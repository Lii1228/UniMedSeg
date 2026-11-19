"""Train or fine-tune either UniMedSeg architecture on grouped image-mask pairs."""

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from eval_script.dataloader import (
    MetaDataset_Multi_Extended,
    MetaDatasetf_transform_1channel,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from UniMedSeg.checkpoints import model_variant, resolve_checkpoint
from UniMedSeg.lightning_model import LightningModel


def gpu_ids(value):
    try:
        ids = [int(part) for part in value.strip("()").split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated GPU indices."
        ) from None
    if not ids or min(ids) < 0 or len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError(
            "GPU indices must be distinct nonnegative integers."
        )
    return ids


def boolean(value):
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--variant", "--model_name", type=model_variant, choices=("S", "L"), default="S"
    )
    p.add_argument(
        "--data_dir", required=True, help="Root containing nnU-Net dataset directories."
    )
    p.add_argument(
        "--stage",
        choices=["stage1", "stage2", "stage2_weaktune"],
        default="stage2_weaktune",
        help="Select the server task-sampling manifest.",
    )
    p.add_argument(
        "--task_config",
        type=Path,
        help="Override the stage manifest with a task list in JSON.",
    )
    checkpoints = p.add_mutually_exclusive_group()
    checkpoints.add_argument(
        "--init",
        "--soft_checkpoint_path",
        help="Initialize model weights from a file or directory.",
    )
    checkpoints.add_argument(
        "--resume",
        "--checkpoint_path",
        help="Resume a full Lightning training checkpoint.",
    )
    p.add_argument(
        "--checkpoint_index",
        type=int,
        default=-1,
        help="Index in lexicographically sorted checkpoint filenames.",
    )
    p.add_argument("--train_gpus", type=gpu_ids, default=[0])
    p.add_argument(
        "--precision",
        choices=["16", "32", "16-mixed", "32-true", "bf16-mixed"],
        default="32",
    )
    p.add_argument("--validation_first", type=int, choices=[0, 1], default=0)
    p.add_argument("--ddp_backend", choices=["gloo", "nccl"], default="gloo")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--batch_size", type=int, choices=[1], default=1, help="One group per device."
    )
    p.add_argument("--context_size", type=int, default=3)
    p.add_argument("--skip_resize", type=boolean, default=True)
    p.add_argument("--len_divid", type=int, default=50)
    p.add_argument(
        "--loss_seg",
        choices=[
            "DiceCE_loss",
            "dice_sigmoid",
            "case_weight",
            "focal_loss",
            "smoothl1",
            "smoothl3_l1",
        ],
        default="DiceCE_loss",
    )
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_decline_patience", type=int, default=120)
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--grad_checkpointing", type=int, choices=[0, 1], default=1)
    p.add_argument(
        "--grad_checkpointing_base",
        type=int,
        choices=[0, 1],
        default=0,
        help="Allow fused attention backends.",
    )
    p.add_argument("--train_aug", type=boolean, default=True)
    p.add_argument("--prob_2d", type=float, default=0.0)
    p.add_argument("--num_slices", type=int, nargs="+", default=[1])
    for strategy in ("a", "c", "ac", "text", "at", "ct", "act"):
        p.add_argument("--prob_" + strategy, type=float, default=float(strategy == "a"))
    p.add_argument(
        "--mode_b_target_prompt",
        choices=["point", "box", "scribble", "lasso", "dense_slice"],
        default="box",
    )
    p.add_argument("--gin_prob", type=float, default=0.05)
    p.add_argument("--sobel_prob", type=float, default=0.0)
    p.add_argument("--flip_prob", type=float, default=0.0)
    p.add_argument("--output_dir", default="runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--limit_train_batches",
        type=int,
        help="Maximum training groups per rank and epoch.",
    )
    p.add_argument(
        "--limit_val_batches",
        type=int,
        help="Maximum validation groups per rank and epoch.",
    )
    args = p.parse_args()
    if min(args.context_size, args.len_divid, args.max_epochs) < 1 or args.workers < 0:
        p.error(
            "Context size, epoch length divisor and epoch count must be positive. Workers must be nonnegative."
        )
    if len(args.num_slices) not in (1, 3) or min(args.num_slices) < 1:
        p.error("Use one slice count or three positive per-axis counts.")
    for name, value in vars(args).items():
        if (name.startswith("prob_") or name.endswith("_prob")) and not 0 <= value <= 1:
            p.error(f"{name} must be a probability in [0, 1].")
    if not any(
        getattr(args, "prob_" + s) for s in ("a", "c", "ac", "text", "at", "ct", "act")
    ):
        p.error("At least one prompt strategy must have positive weight.")
    return args


def make_loaders(args):
    config = args.task_config or Path(__file__).with_name(args.stage + ".json")
    tasks = json.loads(config.read_text(encoding="utf-8"))
    train_transform = (
        MetaDatasetf_transform_1channel(
            flip_prob=args.flip_prob, sobel_prob=args.sobel_prob
        )
        if args.train_aug
        else None
    )
    loaders = []
    for split in ("train", "val"):
        dataset = MetaDataset_Multi_Extended(
            dataset_dir=args.data_dir,
            data_loading_config=tasks,
            train_or_val=split,
            group_size=args.context_size + 1,
            skip_resize=args.skip_resize,
            len_divid=args.len_divid,
            transform=train_transform if split == "train" else None,
        )
        for task, count in zip(tasks, dataset.task_data_num):
            if count == 0:
                raise ValueError(f"{task['name']} has no cases in the {split} split.")
        loaders.append(
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=split == "train",
                num_workers=args.workers,
            )
        )
    return loaders


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    train, val = make_loaders(args)
    model = LightningModel(
        {
            k: v
            for k, v in vars(args).items()
            if k
            in {
                "variant",
                "stage",
                "grad_checkpointing",
                "lr",
                "lr_decline_patience",
                "gin_prob",
                "loss_seg",
                "num_slices",
                "mode_b_target_prompt",
            }
            or k.startswith("prob_")
        }
    )
    source = None
    if args.init or args.resume:
        source = resolve_checkpoint(args.init or args.resume, args.checkpoint_index)
        print(f"Loading UniMedSeg-{args.variant} from {source}", flush=True)
        checkpoint = torch.load(source, map_location="cpu")
        if args.resume and not checkpoint.get("optimizer_states"):
            raise ValueError(
                "Use --init for released weights. Resume requires a full training checkpoint."
            )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
    logger = CSVLogger(args.output_dir, name=f"UniMedSeg-{args.variant}")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.train_gpus,
        strategy=DDPStrategy(
            find_unused_parameters=True, process_group_backend=args.ddp_backend
        )
        if len(args.train_gpus) > 1
        else "auto",
        precision={"16": "16-mixed", "32": "32-true"}.get(
            args.precision, args.precision
        ),
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=1, save_last=True
            )
        ],
    )
    fused = bool(args.grad_checkpointing_base)
    with torch.backends.cuda.sdp_kernel(
        enable_flash=fused, enable_mem_efficient=fused, enable_math=True
    ):
        if args.validation_first:
            trainer.validate(model, dataloaders=val)
        trainer.fit(model, train, val, ckpt_path=str(source) if args.resume else None)


if __name__ == "__main__":
    main()
