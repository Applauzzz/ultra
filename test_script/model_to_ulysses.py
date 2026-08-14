import os
import torch
from torch import distributed as dist
from ultra.model import load_model_from_config

import fla
from fla.layers import (
    Attention,
    HGRN2Attention,
    GatedLinearAttention,
)
from easysp.ulysses_layers import (
    AttentionUlysses,
    HGRN2AttentionUlysses,
    GatedLinearAttentionUlysses,
)

REPLACEMENT_MAP = {
    Attention: AttentionUlysses,
    HGRN2Attention: HGRN2AttentionUlysses,
    GatedLinearAttention: GatedLinearAttentionUlysses,
}

def ulysseslize(model, sp_group):
    for layer in model.model.layers:
        module = layer.attn
        if isinstance(module, Attention):
            new_module = AttentionUlysses(
                hidden_size=module.hidden_size,
                num_heads=module.num_heads,
                num_kv_heads=module.num_kv_heads,
                qkv_bias=module.qkv_bias,
                qk_norm=module.qk_norm,
                window_size=module.window_size,
                rope_theta=module.rope_theta,
                max_position_embeddings=module.max_position_embeddings,
                layer_idx=module.layer_idx,
                sp_group=sp_group,
            )
            setattr(layer, "attn", new_module)
            if dist.get_rank() == 0:
                print(f"Ulysseslize the Attention")
        elif isinstance(module, HGRN2Attention):
            new_module = HGRN2AttentionUlysses(
                mode=module.mode,
                hidden_size=module.hidden_size,
                num_heads=module.num_heads,
                expand_ratio=module.expand_ratio,
                use_short_conv=module.use_short_conv,
                conv_size=module.conv_size,
                conv_bias=module.conv_bias,
                layer_idx=module.layer_idx,
                sp_group=sp_group,
            )
            setattr(layer, "attn", new_module)
            if dist.get_rank() == 0:
                print(f"Ulysseslize the HGRN2Attention")
        elif isinstance(module, GatedLinearAttention):
            new_module = GatedLinearAttentionUlysses(
                mode=module.mode,
                hidden_size=module.hidden_size,
                expand_k=module.expand_k,
                expand_v=module.expand_v,
                num_heads=module.num_heads,
                num_kv_heads=module.num_kv_heads,
                use_short_conv=module.use_short_conv,
                conv_size=module.conv_size,
                conv_bias=module.conv_bias,
                use_output_gate=module.use_output_gate,
                gate_fn=module.gate_fn,
                gate_logit_normalizer=module.gate_logit_normalizer,
                clamp_min=module.clamp_min,
                fuse_norm=False,
                layer_idx=module.layer_idx,
                sp_group=sp_group,
            )
            setattr(layer, "attn", new_module)
            if dist.get_rank() == 0:
                print(f"Ulysseslize the GatedLinearAttention")
        else:
            NotImplementedError(f"Unsupported module {module}")
    if dist.get_rank() == 0:
        print(f"Ulysseslize the Module")


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

if __name__ == "__main__":
    model_name = "hgrn2"
    config_path = "/mnt/bn/tiktok-mm-5/aiic/users/CHOU_Yuhong/codebase/long_context_team/ULTra/main/model_config/1B3_baseline/hybrid71_hgrn2"
    
    rank, world_size, local_rank = setup_distributed()
    sp_group = dist.new_group(ranks=list(range(world_size)))
    torch.manual_seed(42)

    # with torch.device("meta"):
    #     model = load_model_from_config(model_name, config_path)
    # ulysseslize(model, sp_group)
    # model = HGRN2AttentionUlysses(
    #     hidden_size=2048,
    #     sp_group=sp_group,
    # ).to(torch.bfloat16).cuda()
    model = AttentionUlysses(
        hidden_size=2048,
        sp_group=sp_group,
    ).to(torch.bfloat16).cuda()
    hidden_states = torch.randn(1, 2048, 2048).to(torch.bfloat16).cuda().requires_grad_()
    output, _, _ = model(hidden_states)
    output.sum().backward()
    if torch.isnan(hidden_states.grad).any():
        raise ValueError("output is nan")
    if dist.get_rank() == 0:
        print(model)
    # model.to_empty(device="cuda").to(torch.bfloat16)
    # input_ids = torch.randint(0, 32000, (1, 8 * 2048)).to(torch.long).cuda()
    # output = model(input_ids=input_ids, labels=input_ids)
    dist.barrier()
    if dist.get_rank() == 0:
        import pdb
        pdb.set_trace()
    dist.barrier()
    dist.destroy_process_group()


    


