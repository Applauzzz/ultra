# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from copy import deepcopy
import gc
import logging
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf
import torch
import torch.distributed
import torch.nn.functional as F
from torch import distributed as dist
from torch.optim import lr_scheduler
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed._tensor import DTensor

from ultra.args import dataclass_from_dict, dump_config, flatten_dict
from ultra.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint
# ==========
from ultra.data import (
    DataArgs,
    PackTokensState,
    build_dataloader_from_args,
    init_dataloader_state_from_args,
)
# ==========
from ultra.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    dist_mean_dict,
    get_device_mesh,
    get_is_master,
    get_world_size,
    parallelize_model,
    setup_env,
    setup_torch_distributed,
    check_model_value_range,
    sp_dataloader,
)
from ultra.logger import init_logger
from ultra.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)
from ultra.optim import OptimArgs, build_optimizer
from ultra.tokenizer import build_tokenizer
from ultra.model import (
    BaseTransformerArgs,
    reinit_weights,
    reset_rope_cache,
    load_model_from_config,
    get_num_flop_per_token,
    build_fsdp_grouping_plan,
)

import wandb
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

@dataclass
class TrainArgs:
    name: str = "lingua"
    dump_dir: str = ""

    seed: int = 42

    # Number of gradient accumulation steps
    # Total batch size is batch_size*grad_acc_steps
    grad_acc_steps: int = 1

    gc_collect_freq: int = 1000

    # Nb optimizer steps to take
    steps: int = 1000

    data: DataArgs = field(default_factory=DataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: BaseTransformerArgs = field(default_factory=BaseTransformerArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)

    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)

@dataclass
class TrainState(Stateful):
    step: int  # Nb of steps taken by the optimizer
    acc_step: int  # Nb of accumulation steps done since last optimizer step
    scheduler: lr_scheduler.LambdaLR
    data_loader_state: PackTokensState

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "data_loader_state": self.data_loader_state,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.data_loader_state = PackTokensState(**state_dict["data_loader_state"])
        self.scheduler.load_state_dict(state_dict["scheduler"])


def validate_train_args(args: TrainArgs, output_size: int):
    if args.model.vocab_size < 0:
        logger.info(f"Setting model output size to {output_size}")
        args.model.vocab_size = output_size
    assert (
        args.model.vocab_size == output_size
    ), "Vocab size should be the same as output size"

    assert args.dump_dir, "Dump dir not set"

    if args.checkpoint.path is None:
        logger.info(f"Setting checkpoint path to {str(Path(args.dump_dir) / 'checkpoints')}")
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    assert (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        * args.distributed.sp_replicate
        * args.distributed.sp_shard
        == get_world_size()
    ), "DP * TP * SP should be equal to world size"

    if args.distributed.fsdp_type == "no_shard":
        assert (
            args.distributed.dp_shard * args.distributed.sp_shard == 1
            and args.distributed.dp_replicate * args.distributed.sp_replicate == get_world_size()
        )

    args.model.max_seqlen = args.data.seq_len

    # NOTE origin lingua code, no test, kidding?
    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

def verify_world_mesh(
    distributed_args, 
    world_mesh,
):
    dp_mesh = world_mesh["dp_replicate"]
    dp_degree = dp_mesh.size()
    dp_rank = dp_mesh.get_local_rank()
    if distributed_args.dp_shard > 1:
        dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
        dp_degree *= world_mesh["dp_shard"].size()
    
    sp_mesh = world_mesh["sp_replicate"]
    sp_degree = sp_mesh.size()
    sp_rank = sp_mesh.get_local_rank()
    if distributed_args.sp_shard > 1:
        sp_rank = sp_rank * world_mesh["sp_shard"].size() + world_mesh["sp_shard"].get_local_rank()
        sp_degree *= world_mesh["sp_shard"].size()
    
    dp_group = world_mesh["dp"].get_group()
    assert dp_rank == dist.get_rank(group=dp_group)
    sp_group = world_mesh["sp"].get_group()
    assert sp_rank == dist.get_rank(group=sp_group)
    logger.info(f"Running on sp rank : {sp_rank}")

    logger.info(f"Running on dp rank : {dp_rank}")
    logger.info(f"Running on dp size : {dp_degree}")

    # build dataloader
    # need dp world size and rank
    loader_mesh = world_mesh["loader"]
    loader_rank = loader_mesh.get_local_rank()
    loader_degree = loader_mesh.size()

    return dp_rank, dp_degree, sp_rank, sp_degree, loader_rank, loader_degree

def every_n_steps(train_state, freq, acc_step=None, acc_freq=None):
    test = train_state.step % freq == 0
    if acc_step is not None:
        test = test and (train_state.acc_step == acc_step)
    elif acc_freq is not None:
        test = test and ((train_state.acc_step % acc_freq) == 0)
    return test


def train(args: TrainArgs):
    with ExitStack() as context_stack:
        tokenizer = build_tokenizer(args.data.tokenizer.name, args.data.tokenizer.path)
        validate_train_args(
            args,
            tokenizer.n_words,
        )
        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")
        init_logger(Path(args.dump_dir) / "train.log")
        setup_env(args.env)
        setup_torch_distributed(args.distributed)
        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting job: {args.name}")

        dp_rank, dp_degree, sp_rank, sp_degree, loader_rank, loader_degree = verify_world_mesh(args.distributed, world_mesh)

        assert args.data.seq_len % sp_degree == 0, "sp size times of seqlens"

        torch.manual_seed(args.seed)
        logger.info("Building model")

        # Initializing Model in meta device allows us to initialize models much bigger than 1 gpu's memory
        with torch.device("meta"):
            model = load_model_from_config(args.model.model_name, args.model.config_path)
            
        logger.info("Model is built !")

        model_param_count = get_num_params(model)

        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(args.model),
            tp_parallelize=None,
            no_recompute_ops=None,
        )
        model = model.to_empty(device="cuda")

        # NOTE the key point here is that we need to find a distributed checkpoint loading
        if args.checkpoint.init_ckpt_path:
            logger.info(f"Loading initial model from {args.checkpoint.init_ckpt_path}")
            load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model") # Put model_key="" if its directly the model checkpoint
            reset_rope_cache(model) # For RoPe initialization since it's a buffer it might not be loaded
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(args.model.seed)
                reinit_weights(model)
        check_model_value_range(model, range=10.0, std=1.0)

        # log model size

        logger.info(f"Model size: {model_param_count:,} total parameters")

        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # build optimizer after apply parallelisms to the model
        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)
        data_loader_state = init_dataloader_state_from_args(
            args.data, loader_rank, loader_degree
        )

        train_state = TrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)
        # Either load from latest checkpoint or start from scratch
        gc.disable()

        # train loop
        model.train()
        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )
        data_loader = context_stack.enter_context(
            build_dataloader_from_args(
                args.data,
                state=train_state.data_loader_state,
            )
        )

        nwords_since_last_log = 0
        time_last_log = timer()
        gc.collect()
        while train_state.step < args.steps:
            # We constrain train_state.acc_step to be in range 0 to args.grad_acc_steps - 1
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            if every_n_steps(train_state, args.gc_collect_freq, acc_step=0):
                logger.info("garbage collection")
                # we do garbage collection manually otherwise different processes
                # run the GC at different times so they slow down the whole pipeline
                gc.collect()
            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = timer()
            # get batch
            if sp_degree > 1:
                batch, train_state.data_loader_state = sp_dataloader(train_state.step, sp_rank, sp_degree, train_state.data_loader_state, args.data, data_loader)
            else:
                batch, train_state.data_loader_state = next(data_loader)
                batch = torch.tensor(
                    batch,
                    dtype=torch.long,
                    device="cuda",
                ) # bsz seqlen 1
            input_ids = batch[:, :, 0]
            labels = input_ids

            data_load_time = round(timer() - data_load_start, 4)
            nwords_since_last_log += input_ids.numel()

            # forward
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss

            if args.grad_acc_steps > 1:
                model.set_requires_gradient_sync(train_state.acc_step == 0)

            # We scale loss with grad_acc_steps so the gradient is the same
            # regardless of grad_acc_steps
            loss = loss / args.grad_acc_steps
            # backward on scaled loss to create scaled gradients
            loss.backward()
            # For logging we undo that scaling
            loss = loss.detach() * args.grad_acc_steps

            # optimizer step
            grad_norm = -1.0
            if train_state.acc_step == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.optim.clip, foreach=True
                )

                grad_norm = (
                    grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
                ).item()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                train_state.step += 1

            # updates the scale for next iteration
            # training iteration complete
            end_timer.record()

            torch.cuda.synchronize()

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # log metrics
            if every_n_steps(
                train_state,
                args.logging.freq,
                acc_step=None if args.logging.acc_freq else 0,
                acc_freq=args.logging.acc_freq,
            ):
                time_delta = timer() - time_last_log
                tgs = nwords_since_last_log / (time_delta * args.distributed.tp_size * sp_degree)

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                tokens_per_gpu = (
                    total_acc_steps * args.data.batch_size * args.data.seq_len
                )
                total_tokens = dp_degree * tokens_per_gpu
                # This is an estimate and the correct values may change
                # if you change the architecture
                FLOPS = (
                    get_num_flop_per_token(
                        model_param_count - args.model.vocab_size * args.model.dim,
                        args.model.n_layers,
                        args.model.dim,
                        args.data.seq_len,
                    )
                    * tgs
                )
                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "speed": {
                            "tgs": tgs,
                            "FLOPS": FLOPS,
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": data_load_time,
                        },
                        "optim": {
                            "grad_norm": grad_norm,
                            "lr": curr_lr,
                            "total_tokens": total_tokens,
                        },
                        "memory": gpu_mem_stats._asdict(),
                    },
                    sep="/",
                )

                to_sync = {}
                to_sync["loss/out"] = loss.item()
                metrics.update(dist_mean_dict(to_sync))

                if get_is_master():
                    metric_logger.log(metrics)

                gpu_memory_monitor.reset_peak_stats()
                nwords_since_last_log = 0
                time_last_log = timer()
                logger.info(
                    f"step: {train_state.step}"
                    f"  acc: {train_state.acc_step}"
                    f"  loss: {round(loss.item(),4):>7}"
                    f"  grad: {grad_norm:.2e}"
                    f"  flops: {FLOPS:.2e}"
                    f"  tgs: {tgs:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                    f"  pow: {gpu_mem_stats.power_draw/1000} W"
                )

            saved = False
            if every_n_steps(
                train_state, args.checkpoint.dump.every, acc_step=0
            ):
                saved = checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )

    if not saved:
        checkpoint.save(
            model,
            optimizer,
            train_state,
            args,
            device_mesh=world_mesh,
        )
    gc.collect()


def main():
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate TrainArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call train.py with train.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in TrainArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
