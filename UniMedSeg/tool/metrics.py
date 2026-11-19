import torch


def dice_score(prediction, target):
    """Binary Dice at the public evaluation threshold, including empty volumes."""
    predicted = prediction > 0.5
    reference = target > 0.5
    volume = predicted.sum() + reference.sum()
    intersection = (predicted & reference).sum()
    return torch.where(volume > 0, 2 * intersection / volume.clamp_min(1), 1.0)
