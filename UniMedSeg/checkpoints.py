"""Checkpoint selection shared by training and evaluation."""

from pathlib import Path


def model_variant(name):
    variants = {
        "S": "S",
        "UniMedSeg-S": "S",
        "Medverse_prompts_S": "S",
        "UniMedSeg": "S",
        "UniMed_ICL": "S",
        "L": "L",
        "UniMedSeg-L": "L",
        "Medverse_prompts_XL": "L",
    }
    try:
        return variants[name]
    except KeyError:
        raise ValueError(f"Unsupported model {name!r}. Choose S or L.") from None


def resolve_checkpoint(path, index=-1):
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix in {".ckpt", ".pth", ".pt"}
    )
    if not files:
        raise FileNotFoundError(f"No checkpoints in {path}")
    try:
        return files[index]
    except IndexError:
        raise ValueError(
            f"Index {index} is outside the {len(files)} checkpoints in {path}"
        ) from None
