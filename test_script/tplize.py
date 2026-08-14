import os
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed._tensor import DTensor, distribute_module, distribute_tensor
from torch.distributed._tensor.placement_types import Shard, Replicate
from torch.distributed.tensor.parallel import (
    ParallelStyle,
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from fla.models import HGRN2Config, HGRN2ForCausalLM
from functools import partial
from fla.modules.layernorm import rms_norm, RMSNorm
from typing import Any, Optional, Union
from torch.distributed.tensor.placement_types import Placement
from torch.nn import RMSNorm
class RMSNormTP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        
        is_weight_dtensor = isinstance(self.weight, DTensor)
        if is_weight_dtensor:
            weight_local = self.weight.to_local()
        if residual is not None and isinstance(residual, DTensor):
            residual = residual.to_local()
        if isinstance(x, DTensor):
            x = x.to_local()
        
        return rms_norm(
            x,
            weight_local,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )

def from_rms(submodule):
    hidden_size = submodule.hidden_size
    eps = submodule.eps
    elementwise_affine = submodule.elementwise_affine
    bias = False
    return RMSNormTP(
        hidden_size,
        elementwise_affine=elementwise_affine,
        bias=bias,
        eps=eps,
    )

def replace_rmsnorm_with_tp(module: nn.Module) -> None:


    for name, child in list(module.named_children()):
        # 先递归处理子模块
        replace_rmsnorm_with_tp(child)
        
        if isinstance(child, RMSNorm):
            new_module = from_rms(child)
            if child.elementwise_affine:
                new_module.weight = child.weight
                if child.bias is not None:
                    new_module.bias = child.bias
            
            # 替换子模块
            if dist.get_rank() == 0:
                print(f"replace {name} with {new_module}")
            setattr(module, name, new_module)

class RepliParallel(ParallelStyle):
    '''
    the repli is used for modules that needs replicate their parameters

    '''

    def __init__(self,):
        super().__init__()

    def _replicate_module_fn(
        self, name: str, module: nn.Module, device_mesh: DeviceMesh
    ):
        for p_name, param in module.named_parameters():
            # simple replication with fixed ones_ init from LayerNorm/RMSNorm, which allow
            # us to simply just use from_local
            replicated_param = torch.nn.Parameter(
                DTensor.from_local(param, device_mesh, [Replicate()], run_check=False)
            )
            module.register_parameter(p_name, replicated_param)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._replicate_module_fn,
        )

class PrepareModuleInput(ParallelStyle):
    """
    Configure the nn.Module's inputs to convert the input tensors of the nn.Module to DTensors at runtime according to
    ``input_layouts``, and perform layout redistribution according to the ``desired_input_layouts``.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The DTensor layouts of input tensors for the nn.Module, this is used to convert the input tensors to
            DTensors. If some inputs are not torch.Tensor or no need to convert to DTensors, ``None`` need to be specified
            as a placeholder. default: None.
        desired_input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The desired DTensor layout of input tensors for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. This argument needs to have the same length with ``input_layouts``. default: None.
        input_kwarg_layouts (Dict[str, Placement]):
            The DTensor layouts of input kwargs for the nn.Module, this is used to convert the input kwarg tensors to DTensors.
            default: None
        desired_input_kwarg_layouts: (Dict[str, Placement]):
            The desired DTensor layout of input kwargs for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. default: None.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module inputs, default: False.
    Returns:
        A :class:`ParallelStyle` object that prepares the sharding layouts of the nn.Module's inputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the first input of attn will be annotated to Sharded DTensor
        >>> # and then redistributed to Replicated DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan={
        >>>         "attn": PrepareModuleInput(
        >>>             input_layouts=(Shard(0), None, None, ...),
        >>>             desired_input_layouts=(Replicate(), None, None, ...)
        >>>         ),
        >>>     }
        >>> )
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Union[Placement, tuple[Optional[Placement]]]] = None,
        desired_input_layouts: Optional[
            Union[Placement, tuple[Optional[Placement]]]
        ] = None,
        input_kwarg_layouts: Optional[dict[str, Placement]] = None,
        desired_input_kwarg_layouts: Optional[dict[str, Placement]] = None,
        use_local_output: bool = False,
    ):
        self.input_layouts = (
            (input_layouts,) if isinstance(input_layouts, Placement) else input_layouts
        )
        self.desired_input_layouts = (
            (desired_input_layouts,)
            if isinstance(desired_input_layouts, Placement)
            else desired_input_layouts
        )
        self.use_local_output = use_local_output
        if self.input_layouts is not None:
            assert self.desired_input_layouts is not None, (
                "desired module inputs should not be None!"
            )
            assert len(self.input_layouts) == len(self.desired_input_layouts), (
                "input_layouts and desired_input_layouts should have same length!"
            )
        self.with_kwargs = input_kwarg_layouts is not None
        self.input_kwarg_layouts = input_kwarg_layouts or {}
        self.desired_input_kwarg_layouts = desired_input_kwarg_layouts or {}
        if self.with_kwargs:
            assert len(self.input_kwarg_layouts) == len(
                self.desired_input_kwarg_layouts
            ), (
                "input_kwarg_layouts and desired_input_kwarg_layouts should have same length!"
            )

    def _prepare_input_arg(
        self,
        input: Any,
        mesh: DeviceMesh,
        input_layout: Optional[Placement],
        desired_layout: Optional[Placement],
    ):
        if input_layout is not None:
            if isinstance(input, DTensor):
                # TODO: re-enable the check once we fix the compile path
                # assert inp.placements[0] == input_layout
                dt_inp = input
            else:
                assert isinstance(input, torch.Tensor), (
                    "expecting input to be a torch.Tensor!"
                )
                dt_inp = DTensor.from_local(
                    input, mesh, (input_layout,), run_check=False
                )

            if desired_layout is not None and input_layout != desired_layout:
                dt_inp = dt_inp.redistribute(placements=(desired_layout,))

            return dt_inp.to_local() if self.use_local_output else dt_inp
        else:
            return input

    def _prepare_input_fn(self, inputs, device_mesh):
        if self.input_layouts is None:
            return inputs
        prepared_inputs = []
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if len(inputs) != len(self.input_layouts):
            raise ValueError(f"module {inputs} and input_layouts should have same length!")

        assert self.desired_input_layouts is not None, (
            "desired module inputs should not be None!"
        )
        for inp, input_layout, desired_layout in zip(
            inputs, self.input_layouts, self.desired_input_layouts
        ):
            prepared_inputs.append(
                self._prepare_input_arg(inp, device_mesh, input_layout, desired_layout)
            )
        return tuple(prepared_inputs)

    def _prepare_input_kwarg_fn(self, inputs, kwarg_inputs, device_mesh):
        prepared_arg_inputs = self._prepare_input_fn(inputs, device_mesh)
        prepared_kwarg_inputs = {}
        for kwarg_key in kwarg_inputs.keys():
            kwarg_val = kwarg_inputs[kwarg_key]
            input_layout = self.input_kwarg_layouts.get(kwarg_key)
            desired_input_layout = self.desired_input_kwarg_layouts.get(kwarg_key)

            prepared_kwarg_inputs[kwarg_key] = self._prepare_input_arg(
                kwarg_val, device_mesh, input_layout, desired_input_layout
            )

        return (prepared_arg_inputs, prepared_kwarg_inputs)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if self.with_kwargs:
            module.register_forward_pre_hook(
                lambda _, inputs, kwargs: self._prepare_input_kwarg_fn(
                    inputs, kwargs, device_mesh
                ),
                with_kwargs=True,
            )  # type: ignore[misc]
        else:
            module.register_forward_pre_hook(
                lambda _, inputs: self._prepare_input_fn(inputs, device_mesh)
            )  # type: ignore[misc, call-arg]
        return module


# Optional and only used for model/tensor parallelism when tp_size > 1
def tp_parallelize(model, tp_mesh):
    replace_rmsnorm_with_tp(model)

    # Embedding layer tp
    main_plan = {}
    main_plan["model.embeddings"] = ColwiseParallel(
        input_layouts=Replicate(), output_layouts=Shard(1)
    )
    main_plan["model.norm"] = RepliParallel()
    main_plan["lm_head"] = ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Replicate()          
    )

    parallelize_module(
        model,
        tp_mesh,
        main_plan,
    )

    # Attention layers tp
    for layer in model.model.layers:
        
        layer_plan = {}

        layer_plan["attn"] = PrepareModuleInput(
            input_layouts=(Shard(1)),
            desired_input_layouts=(Replicate()),
        )
        layer_plan["attn_norm"] = RepliParallel()
        layer_plan["attn.q_proj"] = ColwiseParallel()
        layer_plan["attn.f_proj"] = ColwiseParallel()
        layer_plan["attn.i_proj"] = ColwiseParallel()
        layer_plan["attn.g_norm"] = RepliParallel()
        layer_plan["attn.o_proj"] = RowwiseParallel(output_layouts=Shard(1))

        # Feedforward layers tp
        layer_plan["mlp"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        layer_plan["mlp_norm"] = RepliParallel()
        layer_plan["mlp.up_proj"] = ColwiseParallel()
        layer_plan["mlp.gate_proj"] = ColwiseParallel()
        layer_plan["mlp.down_proj"] = RowwiseParallel(output_layouts=Shard(1))

        parallelize_module(
            layer,
            tp_mesh,
            layer_plan,
        )

        # Adjusting the number of heads and kv heads according to the tp size
        attn_layer = layer.attn
        attn_layer.n_heads = attn_layer.num_heads // 8

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
    
    rank, world_size, local_rank = setup_distributed()
    mesh = DeviceMesh("cuda", list(range(world_size)))
    config = HGRN2Config.from_pretrained("/main/model_config/1B3_baseline/hgrn2_tp")

    with torch.device("meta"):
        model = HGRN2ForCausalLM(config)
    tp_parallelize(model, mesh)
    model.to_empty(device="cuda")
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(42)
        input_ids = torch.randint(0, 32000, (1, 8 * 2048)).to(torch.long).cuda()
    output = model(input_ids=input_ids)
    dist.barrier()
    if dist.get_rank() == 0:
        print(model)
        print(output.logits.shape)

    dist.destroy_process_group()
