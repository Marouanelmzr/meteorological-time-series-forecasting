import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean",):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none",)

        probs = torch.sigmoid(logits)

        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_factor = torch.where( targets == 1, self.alpha, 1 - self.alpha,)
            focal_weight = alpha_factor * focal_weight

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


class BetaGaussianNLLLoss(nn.Module):
    def __init__(self, beta=0.5, eps=1e-6):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, mean, target, var):
        var = var.clamp(min=self.eps)

        nll = 0.5 * (
            torch.log(var)
            + (target - mean) ** 2 / var
        )

        if self.beta > 0:
            weight = var.detach() ** self.beta
            nll = weight * nll

        return nll.mean()