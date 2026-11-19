"""Arguments shared by the public sample evaluations."""

import argparse


def parse_bool(value):
    if value.lower() in {"true", "1"}:
        return True
    if value.lower() in {"false", "0"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint_path", required=True, help="Checkpoint file or directory."
    )
    parser.add_argument("--checkpoint_index", type=int, default=-1)
    parser.add_argument(
        "--model_name",
        default="UniMedSeg",
        help="Legacy dataset option. The checkpoint selects the architecture.",
    )
    parser.add_argument("--context_size", type=int, default=1)
    parser.add_argument(
        "--task_idx",
        type=int,
        default=0,
        help="Legacy option for the single-task sample manifests.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--skip_resize",
        type=parse_bool,
        default=True,
    )
    parser.add_argument("--len_divid", type=int, default=1)
    parser.add_argument("--train_or_val_split_rate", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, choices=[1], default=1)
    parser.add_argument("--device", default="cuda:0")
    return parser


def evaluation_trainer(device_name):
    import pytorch_lightning as pl
    import torch

    device = torch.device(device_name)
    if device.type == "cuda":
        return pl.Trainer(
            accelerator="gpu",
            devices=[device.index or 0],
            logger=False,
            enable_checkpointing=False,
        )
    if device.type == "cpu":
        return pl.Trainer(
            accelerator="cpu", devices=1, logger=False, enable_checkpointing=False
        )
    raise ValueError(f"Unsupported evaluation device: {device}")
