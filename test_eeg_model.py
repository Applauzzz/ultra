"""
Test script for EEG MAE model.
Verifies that the model can be instantiated and forward pass works.
"""

import sys
from pathlib import Path

# Add ultra to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from ultra.model_eeg import MAEModelArgs, build_mae_model


def test_mae_model():
    """Test MAE model instantiation and forward pass."""
    print("=" * 70)
    print("Testing EEG MAE Model")
    print("=" * 70)

    # Test configuration (Base model)
    config = MAEModelArgs(
        # Encoder
        encoder_embed_dim=512,
        encoder_depth=22,
        encoder_heads=8,
        encoder_head_dim=64,
        encoder_mlp_dim_ratio=2.66,
        encoder_use_geglu=True,

        # Decoder
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_heads=8,
        decoder_head_dim=64,
        decoder_mlp_dim_ratio=2.66,
        decoder_use_geglu=True,

        # Patch config
        patch_size=200,
        patch_overlap=20,
        freqs=4,
        noise_ratio=0.0025,

        # Masking
        masking_ratio=0.75,

        # Token averaging
        token_avg=False,
        token_avg_lambda=0.1,
    )

    print("\nModel Configuration:")
    print(f"  Encoder: {config.encoder_embed_dim}d, {config.encoder_depth} layers")
    print(f"  Decoder: {config.decoder_embed_dim}d, {config.decoder_depth} layers")
    print(f"  Patch size: {config.patch_size}")
    print(f"  Masking ratio: {config.masking_ratio}")

    # Build model
    print("\n" + "-" * 70)
    print("Building model...")
    model = build_mae_model(config)
    print("✓ Model built successfully!")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")

    # Test forward pass
    print("\n" + "-" * 70)
    print("Testing forward pass...")

    # Create dummy input
    batch_size = 2
    n_channels = 19
    time_points = 10000

    eeg = torch.randn(batch_size, n_channels, time_points)
    pos = torch.randn(batch_size, n_channels, 3)  # (x, y, z) positions

    print(f"  Input EEG shape: {eeg.shape}")
    print(f"  Input positions shape: {pos.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        loss = model(eeg, pos)

    print(f"\n✓ Forward pass successful!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss dtype: {loss.dtype}")

    # Test with pre-computed masks
    print("\n" + "-" * 70)
    print("Testing with pre-computed mask indices...")

    # Compute number of patches
    patch_size = config.patch_size
    patch_overlap = config.patch_overlap
    num_patches_per_channel = (time_points - patch_size) // (patch_size - patch_overlap) + 1
    num_patches = n_channels * num_patches_per_channel

    num_masked = int(config.masking_ratio * num_patches)
    num_unmasked = num_patches - num_masked

    # Random mask indices
    mask_indices = torch.randperm(num_patches)[:num_masked].unsqueeze(0).expand(batch_size, -1)
    unmask_indices = torch.randperm(num_patches)[:num_unmasked].unsqueeze(0).expand(batch_size, -1)

    print(f"  Number of patches: {num_patches}")
    print(f"  Masked patches: {num_masked}")
    print(f"  Unmasked patches: {num_unmasked}")

    with torch.no_grad():
        loss = model(eeg, pos, b_m=mask_indices, b_u=unmask_indices)

    print(f"\n✓ Forward pass with pre-computed masks successful!")
    print(f"  Loss: {loss.item():.4f}")

    # Test training mode
    print("\n" + "-" * 70)
    print("Testing training mode...")

    model.train()
    loss = model(eeg, pos)
    loss.backward()

    print(f"✓ Backward pass successful!")
    print(f"  Loss: {loss.item():.4f}")

    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  Parameters with gradients: {has_grad}/{trainable_params}")

    # Test with different batch sizes and channel counts
    print("\n" + "-" * 70)
    print("Testing variable batch sizes and channel counts...")

    test_cases = [
        (1, 8, 10000),
        (4, 19, 10000),
        (2, 64, 10000),
        (1, 128, 10000),
    ]

    model.eval()
    for batch, channels, time in test_cases:
        eeg_test = torch.randn(batch, channels, time)
        pos_test = torch.randn(batch, channels, 3)

        with torch.no_grad():
            loss_test = model(eeg_test, pos_test)

        print(f"  B={batch}, C={channels}, T={time}: loss={loss_test.item():.4f} ✓")

    print("\n" + "=" * 70)
    print("✅ All tests PASSED!")
    print("=" * 70)

    return model


if __name__ == "__main__":
    try:
        test_mae_model()
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ Test FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
