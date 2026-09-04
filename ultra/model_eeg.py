"""
EEG MAE model for ultra framework.
Adapted from eeg_infra/src/models/ to work with ultra's distributed training.
"""

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

import logging

logger = logging.getLogger(__name__)


# Check for FlashAttention availability
try:
    import flash_attn
    FLASH_AVAILABLE = True
except ImportError:
    flash_attn = None
    FLASH_AVAILABLE = False
    logger.info("flash_attn not found, will use PyTorch SDPA instead")


# ==================== Model Configuration ====================

@dataclass
class TransformerConfig:
    """Configuration for Transformer backbone."""
    embed_dim: int = 512
    depth: int = 22
    heads: int = 8
    head_dim: int = 64
    mlp_dim_ratio: float = 2.66
    use_geglu: bool = True


@dataclass
class MAEModelArgs:
    """Configuration for EEG MAE model."""

    # Encoder config
    encoder_embed_dim: int = 512
    encoder_depth: int = 22
    encoder_heads: int = 8
    encoder_head_dim: int = 64
    encoder_mlp_dim_ratio: float = 2.66
    encoder_use_geglu: bool = True

    # Decoder config
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_heads: int = 8
    decoder_head_dim: int = 64
    decoder_mlp_dim_ratio: float = 2.66
    decoder_use_geglu: bool = True

    # Patch config
    patch_size: int = 200
    patch_overlap: int = 20
    freqs: int = 4
    noise_ratio: float = 0.0025

    # Masking config
    masking_ratio: float = 0.75

    # Token averaging (optional auxiliary loss)
    token_avg: bool = False
    token_avg_lambda: float = 0.1

    # Initialization config
    init_std: float = 0.02
    init_cutoff_factor: float = 3.0
    meta_init: bool = True

    @property
    def encoder_config(self) -> TransformerConfig:
        """Get encoder transformer config."""
        return TransformerConfig(
            embed_dim=self.encoder_embed_dim,
            depth=self.encoder_depth,
            heads=self.encoder_heads,
            head_dim=self.encoder_head_dim,
            mlp_dim_ratio=self.encoder_mlp_dim_ratio,
            use_geglu=self.encoder_use_geglu,
        )

    @property
    def decoder_config(self) -> TransformerConfig:
        """Get decoder transformer config."""
        return TransformerConfig(
            embed_dim=self.decoder_embed_dim,
            depth=self.decoder_depth,
            heads=self.decoder_heads,
            head_dim=self.decoder_head_dim,
            mlp_dim_ratio=self.decoder_mlp_dim_ratio,
            use_geglu=self.decoder_use_geglu,
        )


# ==================== Initialization ====================

class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights_megatron(module: nn.Module, std: float, cutoff_factor: float = 3.0) -> None:
    """Initialize weights with truncated normal (Megatron-style)."""
    if isinstance(module, nn.Parameter):
        nn.init.trunc_normal_(
            module,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ==================== Transformer Layers ====================

class GEGLU(nn.Module):
    """Gated Linear Unit with GELU activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.gelu(gates) * x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    """Feed-forward network with optional GEGLU."""

    def __init__(self, dim: int, hidden_dim: int, use_geglu: bool):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, hidden_dim * 2 if use_geglu else hidden_dim, bias=False),
            GEGLU() if use_geglu else nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassicalAttention(nn.Module):
    """Classical multi-head attention with SDPA support."""

    def __init__(self, heads: int, use_sdpa: bool = True):
        super().__init__()
        self.use_sdpa = use_sdpa
        self.heads = heads
        if self.use_sdpa:
            assert version.parse(torch.__version__) >= version.parse("2.2.0"), (
                "SDPA requires PyTorch >= 2.2.0"
            )

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))

        if self.use_sdpa:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = q.shape[-1] ** -0.5
            dots = torch.matmul(q, k.transpose(-1, -2)) * scale
            attn = F.softmax(dots, dim=-1)
            out = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return out


class FlashAttention(nn.Module):
    """FlashAttention wrapper."""

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        if flash_attn is None:
            raise RuntimeError("flash_attn is not available")

        batch_size, seq_len = qkv.shape[:2]
        qkv = rearrange(qkv, "b n (three h d) -> (b n) three h d", three=3, h=self.num_heads)
        cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=qkv.device)

        out = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0.0, causal=False
        )

        out = rearrange(out, "(b n) h d -> b n (h d)", b=batch_size)
        return out


class Attention(nn.Module):
    """Multi-head attention with optional FlashAttention."""

    def __init__(self, dim: int, heads: int = 8, head_dim: int = 64, use_flash: bool = True):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.use_flash = use_flash and FLASH_AVAILABLE
        self.attend: nn.Module = self._get_attend()

    def _get_attend(self) -> nn.Module:
        if self.use_flash:
            return FlashAttention(self.heads)
        else:
            return ClassicalAttention(self.heads, use_sdpa=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FlashAttention requires:
        # 1. CUDA device
        # 2. fp16 or bf16 dtype
        # Fallback to Classical Attention if these conditions are not met
        use_flash_here = (
            self.use_flash and
            x.is_cuda and
            x.dtype in (torch.float16, torch.bfloat16)
        )

        if not use_flash_here and self.use_flash:
            # Temporarily use Classical Attention
            self.attend = ClassicalAttention(self.heads, use_sdpa=True)

        x = self.norm(x)
        qkv = self.to_qkv(x)
        out = self.attend(qkv)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Single Transformer block with attention + FFN."""

    def __init__(self, dim: int, heads: int, head_dim: int, mlp_dim: int, use_geglu: bool):
        super().__init__()
        self.attn = Attention(dim, heads=heads, head_dim=head_dim, use_flash=FLASH_AVAILABLE)
        self.ff = FeedForward(dim, mlp_dim, use_geglu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class TransformerBackbone(nn.Module):
    """Transformer backbone for encoder/decoder."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.dim = config.embed_dim
        mlp_dim = int(config.embed_dim * config.mlp_dim_ratio)

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=config.embed_dim,
                heads=config.heads,
                head_dim=config.head_dim,
                mlp_dim=mlp_dim,
                use_geglu=config.use_geglu,
            )
            for _ in range(config.depth)
        ])

    def forward(self, x: torch.Tensor, return_out_layers: bool = False):
        if return_out_layers:
            out_layers = [x]
            for layer in self.layers:
                x = layer(x)
                out_layers.append(x)
            return out_layers
        else:
            for layer in self.layers:
                x = layer(x)
            return x


# ==================== Positional Embeddings ====================

class FourierEmb4D(nn.Module):
    """Fourier positional embedding for 4D positions (x, y, z, t)."""

    def __init__(self, dimension: int, freqs: int, increment_time: float = 0.1, margin: float = 0.4):
        super().__init__()
        self.dimension = dimension
        self.freqs = freqs
        self.increment_time = increment_time
        self.margin = margin

    def forward(self, positions_: torch.Tensor) -> torch.Tensor:
        positions = positions_.clone()
        positions[:, :, -1] *= self.increment_time
        *U, _ = positions.shape

        freqs_w = torch.arange(self.freqs, device=positions.device, dtype=positions.dtype)
        freqs_z = freqs_w[:, None]
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]

        width = 1 + 2 * self.margin
        positions = positions + self.margin

        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        p_z = 2 * math.pi * freqs_z / width
        p_w = 2 * math.pi * freqs_w / width

        positions = positions[..., None, None, None, None, :]
        loc = (
            positions[..., 0] * p_x +
            positions[..., 1] * p_y +
            positions[..., 2] * p_z +
            positions[..., 3] * p_w
        ).view(*U, -1)

        # Adjust for dimension
        if self.dimension != 512:
            _, _, hd = loc.shape
            diff = hd - self.dimension // 2
            loc = loc[:, :, :-diff]

        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb

    @classmethod
    def add_time_patch(cls, pos: torch.Tensor, num_patches: int) -> torch.Tensor:
        """Add time dimension to positions (B, C, 3) -> (B, C*num_patches, 4)."""
        B, C, _ = pos.shape
        pos_repeated = pos.unsqueeze(2).repeat(1, 1, num_patches, 1)  # (B, C, T, 3)
        time_values = torch.arange(0, num_patches, 1, device=pos.device, dtype=pos.dtype)
        time_values = time_values.view(1, 1, num_patches, 1).expand(B, C, num_patches, 1)
        pos_with_time = torch.cat((pos_repeated, time_values), dim=-1)  # (B, C, T, 4)
        pos_with_time = pos_with_time.view(B, C * num_patches, 4)
        return pos_with_time


def patch_embedding(embed_dim: int, patch_size: int) -> nn.Module:
    """Linear projection for patches."""
    return nn.Sequential(nn.Linear(patch_size, embed_dim))


def mlp_pos_embedding(embed_dim: int) -> nn.Module:
    """MLP for positional embeddings."""
    return nn.Sequential(
        nn.Linear(4, embed_dim, bias=False),
        nn.GELU(),
        nn.LayerNorm(embed_dim),
    )


# ==================== REVE Encoder ====================

class REVE(nn.Module):
    """REVE encoder for EEG signals."""

    def __init__(self, config: MAEModelArgs):
        super().__init__()

        self.transformer = TransformerBackbone(config.encoder_config)
        self.embed_dim = config.encoder_embed_dim

        self.freqs = config.freqs
        self.patch_size = config.patch_size
        self.overlap_size = config.patch_overlap
        self.noise_ratio = config.noise_ratio

        self.to_patch_embedding = patch_embedding(self.embed_dim, config.patch_size)
        self.fourier4d = FourierEmb4D(self.embed_dim, freqs=config.freqs)
        self.mlp4d = mlp_pos_embedding(self.embed_dim)
        self.ln = nn.LayerNorm(self.embed_dim)

    def forward(self, eeg: torch.Tensor, pos: Optional[torch.Tensor] = None, return_output: bool = False):
        device = eeg.device
        eeg = eeg.float()

        # Patchify
        patches = eeg.unfold(dimension=2, size=self.patch_size, step=self.patch_size - self.overlap_size)
        b, c, h, p = patches.shape

        # Add noise to positions during training
        if self.training and pos is not None:
            noise = torch.from_numpy(
                np.random.normal(loc=0, scale=self.noise_ratio, size=(c, 3))
            ).to(device).to(pos.dtype)
            pos = pos + noise

        # Positional embeddings
        pos = FourierEmb4D.add_time_patch(pos, h)
        pos_embed = self.ln(self.fourier4d(pos) + self.mlp4d(pos))

        # Patch embeddings + positional embeddings
        x = rearrange(
            self.to_patch_embedding(patches),
            "b c h e -> b (c h) e",
            c=c, h=h, e=self.embed_dim
        ) + pos_embed

        # Transformer
        x = self.transformer(x, return_output)
        return x


# ==================== MAE Model ====================

class MAE(nn.Module):
    """Masked Autoencoder for EEG pretraining."""

    def __init__(self, config: MAEModelArgs):
        super().__init__()
        self.config = config

        self.masking_ratio = config.masking_ratio
        assert 0 < self.masking_ratio < 1, "masking_ratio must be in (0, 1)"

        # Encoder
        self.encoder = REVE(config)

        encoder_dim = self.encoder.embed_dim
        pixel_values_per_patch = config.patch_size

        # Decoder
        self.decoder = TransformerBackbone(config.decoder_config)

        decoder_dim = config.decoder_embed_dim
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))

        # Encoder-to-decoder projection
        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )
        self.pos_enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )

        # Decoder-to-pixel projection
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

        # Token averaging (optional)
        self.token_avg = config.token_avg
        if self.token_avg:
            self.cls_query_token = nn.Parameter(torch.randn(1, 1, encoder_dim))
            self.cls_to_pixels = nn.Sequential(
                nn.Linear(encoder_dim, 4 * encoder_dim, bias=False),
                nn.ReLU(),
                nn.Linear(4 * encoder_dim, pixel_values_per_patch),
            )
            self.token_avg_lambda = config.token_avg_lambda

        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        std = self.config.init_std
        cutoff = self.config.init_cutoff_factor

        # Initialize mask token
        init_weights_megatron(self.mask_token, std, cutoff)

        # Initialize token averaging
        if self.token_avg:
            init_weights_megatron(self.cls_query_token, std, cutoff)

        # Initialize linear layers in transformer
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "encoder.transformer.layers" in name or "decoder.layers" in name:
                    init_weights_megatron(module, std, cutoff)

        logger.info("MAE weights initialized with Megatron-style init")

    def forward(
        self,
        eeg: torch.Tensor,
        pos: torch.Tensor,
        b_m: Optional[torch.Tensor] = None,
        b_u: Optional[torch.Tensor] = None,
        return_patches: bool = False,
    ):
        """
        Forward pass for MAE training.

        Args:
            eeg: (B, C, T) EEG signals
            pos: (B, C, 3) electrode positions
            b_m: (B, num_masked) masked indices (optional, auto-generated if None)
            b_u: (B, num_unmasked) unmasked indices (optional)
            return_patches: whether to return predicted and ground-truth patches

        Returns:
            loss: reconstruction loss
            (optional) pred_pixel_values, masked_patches
        """
        device = eeg.device

        # Patchify
        patches = eeg.unfold(
            dimension=2,
            size=self.encoder.patch_size,
            step=self.encoder.patch_size - self.encoder.overlap_size,
        )
        b, c, h, p = patches.shape
        patches = rearrange(patches, "b c h e -> b (c h) e", c=c, h=h, e=p)
        batch = b
        num_patches = c * h

        # Add noise to positions during training
        if self.training:
            noise = torch.from_numpy(
                np.random.normal(loc=0, scale=self.encoder.noise_ratio, size=(c, 3))
            ).to(device).to(pos.dtype)
            pos = pos + noise

        # Positional embeddings
        pos = FourierEmb4D.add_time_patch(pos, h)
        pos_embed = self.encoder.ln(self.encoder.fourier4d(pos) + self.encoder.mlp4d(pos))
        tokens = self.encoder.to_patch_embedding(patches) + pos_embed

        # Generate mask indices
        if b_m is None:
            num_masked = int(self.masking_ratio * num_patches)
            if self.training:
                rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
            else:
                torch.manual_seed(42)
                rand_indices = torch.rand(num_patches, device=device).unsqueeze(0).repeat(batch, 1).argsort(dim=-1)
            masked_indices = rand_indices[:, :num_masked]
            unmasked_indices = rand_indices[:, num_masked:]
        else:
            num_masked = b_m.shape[1]
            masked_indices = b_m
            unmasked_indices = b_u

        # Get unmasked tokens
        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # Get masked patches (for loss)
        masked_patches = patches[batch_range, masked_indices]

        # Encode
        if self.token_avg and self.training:
            all_outputs = self.encoder.transformer(tokens, return_out_layers=True)
            encoded_tokens = all_outputs[-1]
            x = torch.cat(all_outputs, dim=1)
            b_size = x.shape[0]

            # Attention pooling
            query_output = self.cls_query_token.expand(b_size, -1, -1)
            key_value_tokens = x
            attention_scores = torch.matmul(
                query_output, key_value_tokens.transpose(-1, -2)
            ) / (self.encoder.embed_dim ** 0.5)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            context = torch.matmul(attention_weights, key_value_tokens).squeeze(1)
        else:
            encoded_tokens = self.encoder.transformer(tokens)
            context = None

        # Project encoder to decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # Decoder position embeddings
        decoder_pos_emb = self.pos_enc_to_dec(pos_embed.reshape(b, c * h, -1)).to(pos_embed)
        unmasked_decoder_tokens = decoder_tokens + decoder_pos_emb[batch_range, unmasked_indices]

        # Mask tokens
        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + decoder_pos_emb[batch_range, masked_indices]

        # Combine masked and unmasked tokens
        decoder_tokens_full = torch.zeros(batch, num_patches, self.decoder.dim, device=device, dtype=tokens.dtype)
        decoder_tokens_full[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens_full[batch_range, masked_indices] = mask_tokens

        # Decode
        decoded_tokens = self.decoder(decoder_tokens_full)

        # Get predictions for masked tokens
        mask_tokens_decoded = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens_decoded)

        # Reconstruction loss
        loss = F.l1_loss(pred_pixel_values, masked_patches)

        # Token averaging auxiliary loss
        if self.token_avg and self.training and context is not None:
            repeated_context = (
                repeat(context, "b d -> b n d", n=num_masked) +
                decoder_pos_emb[batch_range, masked_indices]
            )
            cls_pixels = self.cls_to_pixels(repeated_context)
            loss_cls = F.l1_loss(cls_pixels, masked_patches)
            loss = loss + self.token_avg_lambda * loss_cls

        if return_patches:
            return loss, pred_pixel_values, masked_patches
        else:
            return loss


# ==================== FSDP Support ====================

def build_fsdp_grouping_plan_eeg(args: MAEModelArgs):
    """
    Define FSDP wrapping strategy for MAE model.

    Returns a list of (module_path, reshard_after_forward) tuples.
    This ensures FSDP shards parameters efficiently.
    """
    group_plan = []

    # Wrap each encoder layer
    for i in range(args.encoder_depth):
        group_plan.append((f"encoder.transformer.layers.{i}", True))

    # Wrap each decoder layer
    for i in range(args.decoder_depth):
        group_plan.append((f"decoder.layers.{i}", True))

    return group_plan


# ==================== Model Builder ====================

def build_mae_model(args: MAEModelArgs) -> MAE:
    """
    Build MAE model from configuration.

    Args:
        args: Model configuration

    Returns:
        MAE model instance
    """
    model = MAE(args)
    logger.info(f"Built MAE model: "
                f"encoder_dim={args.encoder_embed_dim}, "
                f"encoder_depth={args.encoder_depth}, "
                f"decoder_dim={args.decoder_embed_dim}, "
                f"decoder_depth={args.decoder_depth}")
    return model
