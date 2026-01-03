import argparse
import platform
import os
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp.grad_scaler import GradScaler
import torch._functorch.config as functorch_config

import yaml
from yaml import CLoader

from ml_suite.models.model_utils import get_model
from ml_suite.models.loss_fns import MAE, MSE, RMSE, NRMSE, VRMSE

from ml_suite.data.dataloader import get_dataloader
from ml_suite.data.dataset import get_dataset

from ml_suite.train.train_base import Trainer
from ml_suite.train.utils.optimizer import get_optimizer
from ml_suite.train.utils.lr_scheduler import get_lr_scheduler
from ml_suite.train.utils.checkpoint_utils import load_checkpoint
from ml_suite.train.utils.wandb_logger import WandbLogger
from ml_suite.train.utils.logger import setup_logger


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=CLoader)
    return config


def time_str_to_seconds(time_str: str) -> float:
    return sum(x * int(t) for x, t in zip([3600, 60, 1], time_str.split(":")))


def get_checkpoint_path(output_dir: Path, checkpoint_name: str) -> Path:
    if checkpoint_name == "latest":
        checkpoint_path = output_dir / "latest.pt"
    elif checkpoint_name == "best":
        checkpoint_path = output_dir / "best.pt"
    elif checkpoint_name.isdigit():
        checkpoint_path = output_dir / f"epoch_{checkpoint_name}/checkpoint.pt"
    else:
        raise ValueError(f"Invalid checkpoint name: {checkpoint_name}")

    return checkpoint_path


@record
def main(
    config_path: Path,
):
    load_dotenv()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl")

    logger = setup_logger("Startup", rank=global_rank)

    config = load_config(config_path)
    output_dir = config_path.parent

    time_limit = config.get("time_limit", None)
    if time_limit is not None:
        time_limit = time_str_to_seconds(time_limit)

    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])

    total_updates = int(
        float(config["total_updates"])
    )  # first float to allow yaml scientific notation
    updates_per_epoch = int(float(config["updates_per_epoch"]))
    cp_every_updates = int(float(config["checkpoint_every_updates"]))
    eval_fraction = float(config.get("eval_fraction", 1.0))

    if global_rank == 0:
        wandb_logger = WandbLogger(
            config["wandb"], log_dir=output_dir, rank=global_rank
        )
        wandb_logger.update_config(config)  # Log full config with defaults
    else:
        wandb_logger = None

    samples_trained = 0
    batches_trained = 0
    epoch = 1

    ############################################################
    ###### AMP #################################################
    ############################################################
    use_amp = config.get("use_amp", True)
    amp_precision_str = config.get("amp_precision", "bfloat16")
    if amp_precision_str == "bfloat16":
        amp_precision = torch.bfloat16
    elif amp_precision_str == "float16":
        amp_precision = torch.float16
    else:
        print(f"Unknown amp_precision {amp_precision_str}, turing off AMP")
        use_amp = False
        amp_precision = torch.float32

    max_grad_norm = config.get("max_grad_norm", None)
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)

    scaler = GradScaler(device=str(device), enabled=use_amp)

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ############################################################
    ###### Load datasets and dataloaders #######################
    ############################################################

    dataset_train = get_dataset(config["dataset"], split="train")
    dataset_val = get_dataset(config["dataset"], split="valid")

    train_dataloader = get_dataloader(
        dataset=dataset_train,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        is_distributed=dist.is_initialized(),
        shuffle=True,
    )
    val_dataloader = get_dataloader(
        dataset=dataset_val,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        is_distributed=dist.is_initialized(),
        shuffle=False,
    )

    ############################################################
    ###### Load torch modules ##################################
    ############################################################

    model = get_model(config["model"])
    model.to(device)

    criterion = config["model"].get("criterion", "MSE")
    if criterion.lower() == "mse":
        criterion_fn = MSE()
    elif criterion.lower() == "mae":
        criterion_fn = MAE()
    else:
        raise ValueError(f"Unknown criterion {criterion}")

    # these are used for evaluation during training (Wandb logging)
    # these are NOT the loss functions used for training (see criterion)
    eval_loss_fns = {
        "MSE": MSE(),
        "MAE": MAE(),
        "RMSE": RMSE(),
        "NRMSE": NRMSE(),
        "VRMSE": VRMSE(),
    }

    ############################################################
    ###### Load checkpoint #####################################
    ############################################################
    checkpoint: Optional[dict] = None
    cp_config: dict = config.get("checkpoint", {})
    checkpoint_name: Optional[str] = cp_config.get("checkpoint_name", None)

    if checkpoint_name is not None:
        logger.info(f"Loading checkpoint: {checkpoint_name}")
        checkpoint_path = get_checkpoint_path(output_dir, checkpoint_name)

        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path, device)
        else:
            logger.warning(
                f"Checkpoint {checkpoint_path} not found, starting from scratch"
            )

    ############################################################
    ###### Load model weights ##################################
    ############################################################
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    ############################################################
    ###### Compile and distribute model #########################
    ############################################################
    functorch_config.activation_memory_budget = config.get("mem_budget", 1)
    compile_model = config.get("compile", False)
    if compile_model and not platform.system() == "Windows":
        model = torch.compile(model, mode="max-autotune")
        logger.info("Model compiled with torch.compile")
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=device,
        )
        logger.info("Model wrapped with DDP")

    if wandb_logger is not None:
        wandb_logger.watch(model, criterion=criterion_fn)

    ############################################################
    ###### Setup optimizers and lr schedulers ##################
    ############################################################
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    lr_config: Optional[dict] = config.get("lr_scheduler", None)
    optimizer = get_optimizer(model, config["optimizer"])  # type: ignore

    restart = cp_config.get("restart", False)
    if checkpoint is not None and restart:
        samples_trained = checkpoint["samples_trained"]
        batches_trained = checkpoint["batches_trained"]
        epoch = checkpoint["epoch"]

        # Create scheduler BEFORE loading optimizer state dict, so it captures
        # the correct base LR (used for eta_min in cosine annealing, etc.)
        if lr_config is not None:
            lr_scheduler = get_lr_scheduler(
                optimizer,
                lr_config,
                total_batches=total_updates,
                total_batches_trained=0,  # Initial state before loading
            )
            # Load state dicts AFTER scheduler creation
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        grad_scaler_sd = checkpoint.get("grad_scaler_state_dict", None)
        if grad_scaler_sd is not None:
            scaler.load_state_dict(grad_scaler_sd)

    else:
        if lr_config is not None:
            lr_scheduler = get_lr_scheduler(
                optimizer,
                lr_config,
                total_batches=total_updates,
                total_batches_trained=batches_trained,
            )

    ############################################################
    ###### Initialize trainer ##################################
    ############################################################

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion_fn,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        scaler=scaler,
        total_updates=total_updates,
        updates_per_epoch=updates_per_epoch,
        checkpoint_every_updates=cp_every_updates,
        eval_fraction=eval_fraction,
        epoch=epoch,
        batches_trained=batches_trained,
        samples_trained=samples_trained,
        loss_fns=eval_loss_fns,
        amp=use_amp,
        amp_precision=amp_precision,
        max_grad_norm=max_grad_norm,
        output_dir=output_dir,
        wandb_logger=wandb_logger,
        time_limit=time_limit,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
    )
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(args.config_path)

    main(config_path)
