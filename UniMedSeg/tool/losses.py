import torch
from torch import nn
from torch.nn import functional as F


def sigmoid_modified(pred, scale=8, offset=0.5):
    return torch.sigmoid(scale * (pred - offset))


def l1_loss(outputs, targets, weights=1.0):
    return torch.mean(abs(outputs - targets) * weights)


def l2_loss(outputs, targets, weights=1.0):
    return torch.mean((outputs - targets) ** 2 * weights)


class GradientLoss(nn.Module):
    def __init__(self, mode="smoothl1", mask=False):
        super(GradientLoss, self).__init__()
        self.mask = mask
        if mode == "l1":
            self.loss_func = l1_loss
        elif mode == "l2":
            self.loss_func = l2_loss
        elif mode == "smoothl1":
            self.loss_func = F.smooth_l1_loss
        else:
            raise ValueError("Not supported loss_func for GradientLoss:", mode)

    def gradient(self, x):
        # x: (b, c, s, r, c) -->  dx, dy, dz: (b, c, s, r, c)
        back = F.pad(x, [0, 1, 0, 0, 0, 0])[:, :, :, :, 1:]
        right = F.pad(x, [0, 0, 0, 1, 0, 0])[:, :, :, 1:, :]
        bottom = F.pad(x, [0, 0, 0, 0, 0, 1])[:, :, 1:, :, :]

        dx, dy, dz = back - x, right - x, bottom - x
        dx[:, :, :, :, -1] = 0
        dy[:, :, :, -1] = 0
        dz[:, :, -1] = 0
        return dx, dy, dz

    def forward_archive(self, input, target):
        dx_i, dy_i, dz_i = self.gradient(input)
        dx_t, dy_t, dz_t = self.gradient(target)
        if self.mask:
            dx_i[dx_t == 0.0] = 0.0
            dy_i[dy_t == 0.0] = 0.0
            dz_i[dz_t == 0.0] = 0.0
        return (
            self.loss_func(dx_i, dx_t)
            + self.loss_func(dy_i, dy_t)
            + self.loss_func(dz_i, dz_t)
        ).mean()

    def forward(self, input, target, weights=1.0):
        dx_i, dy_i, dz_i = self.gradient(input)
        dx_t, dy_t, dz_t = self.gradient(target)
        if self.mask:
            diff_dx = abs(dx_i - dx_t)
            diff_dy = abs(dy_i - dy_t)
            diff_dz = abs(dz_i - dz_t)
            diff_dx[target == 0.0] = 0.0
            diff_dy[target == 0.0] = 0.0
            diff_dz[target == 0.0] = 0.0
            return (diff_dx + diff_dy + diff_dz).mean()
        return (
            (
                self.loss_func(dx_i, dx_t, weights)
                + self.loss_func(dy_i, dy_t, weights)
                + self.loss_func(dz_i, dz_t, weights)
            )
            / 3
        ).mean()


class SmoothL3_L1Loss(nn.Module):
    def __init__(self, beta=1.0, reduction="mean"):
        super(SmoothL3_L1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, input, target):
        n = torch.abs(input - target)
        cond = n < self.beta
        loss = torch.where(
            cond, 0.333 * n**3 / self.beta**2, n + 0.333 * self.beta**3 - self.beta
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
