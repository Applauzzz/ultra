"""
EEG MAE Training Script for ultra framework.
Supports distributed training with FSDP, checkpoint management, and WandB logging.
"""

import gc
import logging
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed._tensor import DTensor

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultra.args import dataclass_from_dict, dump_config, flatten_dict
from ultra.checkpoint import CheckpointArgs, CheckpointManager, load_from_checkpoint
from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader, EEGDataLoaderState
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
)
from ultra.logger import init_logger
from ultra.metrics import (
    GPUMemoryMonitor,
    LoggingArgs,
    MetricLogger,
    get_num_params,
)
from ultra.optim import OptimArgs, build_optimizer
from ultra.model_eeg import MAEModelArgs, build_mae_model, build_fsdp_grouping_plan_eeg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


@dataclass
class EEGTrainArgs:
    """Training arguments for EEG MAE."""

    name: str = "eeg_mae"
    dump_dir: str = ""

    seed: int = 42

    # Number of gradient accumulation steps
    grad_acc_steps: int = 1

    gc_collect_freq: int = 1000

    # Number of optimizer steps to take
    steps: int = 10000

    # Configuration components
    data: EEGDataArgs = field(default_factory=EEGDataArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    model: MAEModelArgs = field(default_factory=MAEModelArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)
    checkpoint: CheckpointArgs = field(default_factory=CheckpointArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)


@dataclass
class EEGTrainState:
    """Training state for EEG MAE."""

    step: int  # Number of optimizer steps
    acc_step: int  # Number of accumulation steps since last optimizer step
    scheduler: torch.optim.lr_scheduler.LambdaLR
    data_loader_state: EEGDataLoaderState

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "data_loader_state": self.data_loader_state.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.data_loader_state.load_state_dict(state_dict["data_loader_state"])
        self.scheduler.load_state_dict(state_dict["scheduler"])


def validate_train_args(args: EEGTrainArgs):
    """Validate and set default training arguments."""

    assert args.dump_dir, "Dump dir not set"

    if args.checkpoint.path is None:
        logger.info(f"Setting checkpoint path to {str(Path(args.dump_dir) / 'checkpoints')}")
        args.checkpoint.path = str(Path(args.dump_dir) / "checkpoints")

    # Validate distributed args
    assert (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        * args.distributed.sp_size
        == get_world_size()
    ), f"DP * TP * SP should equal world size but got {args.distributed.dp_replicate} * {args.distributed.dp_shard} * {args.distributed.tp_size} * {args.distributed.sp_size} != {get_world_size()}"

    if args.distributed.fsdp_type == "no_shard":
        assert (
            args.distributed.dp_shard * args.distributed.sp_size == 1
            and args.distributed.dp_replicate * args.distributed.sp_size == get_world_size()
        )

    # Set WandB name
    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name


def train(args: EEGTrainArgs):
    """Main training function."""

    with ExitStack() as context_stack:
        # Validate arguments
        validate_train_args(args)

        # Setup directories and logging
        if get_is_master():
            os.makedirs(args.dump_dir, exist_ok=True)
            dump_config(args, Path(args.dump_dir) / "config.yaml")

        init_logger(Path(args.dump_dir) / "train.log")
        setup_env(args.env)
        setup_torch_distributed(args.distributed)

        world_mesh = get_device_mesh(args.distributed)
        logger.info(f"Starting EEG MAE training job: {args.name}")

        # Get distributed ranks
        dp_mesh = world_mesh["dp_replicate"]
        dp_degree = dp_mesh.size()
        dp_rank = dp_mesh.get_local_rank()

        if args.distributed.dp_shard > 1:
            dp_rank = dp_rank * world_mesh["dp_shard"].size() + world_mesh["dp_shard"].get_local_rank()
            dp_degree *= world_mesh["dp_shard"].size()

        loader_mesh = world_mesh["loader"]
        loader_rank = loader_mesh.get_local_rank()
        loader_degree = loader_mesh.size()

        logger.info(f"DP rank: {dp_rank}/{dp_degree}")
        logger.info(f"Loader rank: {loader_rank}/{loader_degree}")

        # Set random seed
        torch.manual_seed(args.seed)

        # ==================== Build Model ====================
        logger.info("Building EEG MAE model")

        if args.model.meta_init:
            with torch.device("meta"):
                model = build_mae_model(args.model)
        else:
            model = build_mae_model(args.model).to("cuda")

        logger.info("Model built successfully")

        # Get parameter count before parallelization
        model_param_count = get_num_params(model)

        # Parallelize model with FSDP
        model = parallelize_model(
            model,
            world_mesh,
            args.model,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan_eeg(args.model),
            tp_parallelize=None,
            no_recompute_ops=None,
        )

        if args.model.meta_init:
            model = model.to_empty(device="cuda")

        # Load from checkpoint if specified
        if args.checkpoint.init_ckpt_path and args.model.meta_init:
            logger.info(f"Loading initial model from {args.checkpoint.init_ckpt_path}")
            load_from_checkpoint(args.checkpoint.init_ckpt_path, model, model_key="model")
        elif args.model.meta_init:
            # Model is already initialized in build_mae_model
            logger.info("Using Megatron-style weight initialization")

        check_model_value_range(model, range=10.0, std=1.0)

        logger.info(f"Model size: {model_param_count:,} total parameters")

        # GPU memory monitor
        gpu_memory_monitor = GPUMemoryMonitor("cuda")
        logger.info(
            f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
            f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
        )
        logger.info(f"GPU memory usage: {gpu_memory_monitor}")

        # ==================== Build Optimizer ====================
        optimizer, scheduler = build_optimizer(model, args.optim, args.steps)

        # ==================== Build DataLoader ====================
        logger.info("Building EEG DataLoader")

        # Set distributed args for dataloader
        args.data.world_size = loader_degree
        args.data.rank = loader_rank

        data_loader, data_loader_state = build_eeg_dataloader(
            args.data,
            train=True,
            state=None,
        )

        logger.info(f"DataLoader built: {len(data_loader)} batches")

        # ==================== Training State ====================
        train_state = EEGTrainState(
            step=0,
            acc_step=0,
            data_loader_state=data_loader_state,
            scheduler=scheduler,
        )

        # ==================== Checkpoint Manager ====================
        checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
        checkpoint.load(model, optimizer, train_state, world_mesh)

        # Disable automatic garbage collection for performance
        gc.disable()

        # ==================== Training Loop ====================
        model.train()

        metric_logger = context_stack.enter_context(
            MetricLogger(Path(args.dump_dir) / "metrics.jsonl", args)
        )

        if get_is_master():
            # Record total parameters
            metric_logger.summary("model_param_count", model_param_count)

            # Record model architecture
            model_structure_str = str(model)
            with open(Path(args.dump_dir) / "model_structure.txt", "w") as f:
                f.write(model_structure_str)
            metric_logger.save(str(Path(args.dump_dir) / "model_structure.txt"))

        # Training metrics
        nwords_since_last_log = 0
        time_last_log = time.time()
        gc.collect()

        # Convert DataLoader to iterator
        data_loader = iter(data_loader)

        logger.info("Starting training loop")

        while train_state.step < args.steps:
            # Update accumulation step
            train_state.acc_step += 1
            train_state.acc_step = train_state.acc_step % args.grad_acc_steps

            # Garbage collection
            if train_state.step % args.gc_collect_freq == 0 and train_state.acc_step == 0:
                logger.info("Running garbage collection")
                gc.collect()

            curr_lr = float(optimizer.param_groups[0]["lr"])
            data_load_start = time.time()

            # Get batch
            try:
                eeg, pos, batch_mask, batch_unmask = next(data_loader)
            except StopIteration:
                # Restart data loader
                data_loader = iter(
                    context_stack.enter_context(
                        build_eeg_dataloader(
                            args.data,
                            train=True,
                            state=train_state.data_loader_state,
                        )[0]
                    )
                )
                eeg, pos, batch_mask, batch_unmask = next(data_loader)

            # Move to CUDA
            eeg = eeg.to("cuda")
            pos = pos.to("cuda")
            batch_mask = batch_mask.to("cuda")
            batch_unmask = batch_unmask.to("cuda")

            data_load_time = round(time.time() - data_load_start, 4)
            nwords_since_last_log += eeg.numel()

            # Forward pass
            start_timer = torch.cuda.Event(enable_timing=True)
            end_timer = torch.cuda.Event(enable_timing=True)
            start_timer.record()

            loss = model(eeg, pos, b_m=batch_mask, b_u=batch_unmask)

            # Scale loss for gradient accumulation
            loss = loss / args.grad_acc_steps

            # Backward pass
            loss.backward()

            # Unscale loss for logging
            loss = loss.detach() * args.grad_acc_steps

            # Optimizer step
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

            end_timer.record()
            torch.cuda.synchronize()

            curr_iter_time = round(start_timer.elapsed_time(end_timer) * 1e-3, 4)

            # Logging
            if train_state.step % args.logging.freq == 0 and train_state.acc_step == 0:
                time_delta = time.time() - time_last_log

                # Tokens per second (treating each EEG sample as a "token")
                tps = nwords_since_last_log / (time_delta * args.distributed.tp_size)

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                total_acc_steps = (
                    args.grad_acc_steps * train_state.step + train_state.acc_step
                )
                samples_per_gpu = total_acc_steps * args.data.batch_size
                total_samples = dp_degree * samples_per_gpu

                metrics = flatten_dict(
                    {
                        "global_step": train_state.step,
                        "acc_step": train_state.acc_step,
                        "speed": {
                            "samples_per_sec": tps,
                            "curr_iter_time": curr_iter_time,
                            "data_load_time": data_load_time,
                        },
                        "optim": {
                            "grad_norm": grad_norm,
                            "lr": curr_lr,
                            "total_samples": total_samples,
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
                time_last_log = time.time()

                logger.info(
                    f"step: {train_state.step}"
                    f"  acc: {train_state.acc_step}"
                    f"  loss: {round(loss.item(), 4):>7}"
                    f"  grad: {grad_norm:.2e}"
                    f"  sps: {tps:.2e}"
                    f"  iter: {curr_iter_time:>7}"
                    f"  data: {data_load_time:>5}"
                    f"  lr: {curr_lr:.2e}"
                    f"  mem: {gpu_mem_stats.max_active_pct:.0f}%"
                )

            # Checkpoint saving
            if train_state.step % args.checkpoint.dump.every == 0 and train_state.acc_step == 0:
                checkpoint.save(
                    model,
                    optimizer,
                    train_state,
                    args,
                    device_mesh=world_mesh,
                )

        # Final checkpoint
        checkpoint.save(
            model,
            optimizer,
            train_state,
            args,
            device_mesh=world_mesh,
        )

        gc.collect()
        logger.info("Training complete!")


def main():
    """
    Command line interface using OmegaConf.

    Usage:
        python main/train_eeg.py config=configs/eeg/base.yaml name=my_experiment

    The behavior is:
    1. Instantiate EEGTrainArgs with default values
    2. Override with values from config file
    3. Override with additional command line arguments
    """
    cli_args = OmegaConf.from_cli()

    # Load config file if specified
    if "config" in cli_args:
        file_cfg = OmegaConf.load(cli_args.config)
        del cli_args.config
    else:
        file_cfg = OmegaConf.create()

    # Merge: defaults < file config < CLI args
    default_cfg = OmegaConf.structured(EEGTrainArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    train(cfg)


if __name__ == "__main__":
    main()
