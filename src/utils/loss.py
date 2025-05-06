import torch

from torch import nn, einsum
from torchvision.ops import sigmoid_focal_loss
from typing import Literal


class DiceLoss(nn.Module):
    def __init__(self,
                 epsilon: float = 1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
            
    def forward_loss(self, inputs, targets):
        intersection = einsum("bcwh,bcwh->bc", inputs, targets)
        sum_probs = einsum("bcwh->bc", inputs) + einsum("bcwh->bc", targets)
        loss = (2. * intersection + self.epsilon) / (sum_probs + self.epsilon)  

        return 1 - loss
    
    def forward(self, inputs, targets):
        inputs = nn.functional.sigmoid(inputs)
        loss = self.forward_loss(inputs, targets)
        loss = torch.mean(loss)
        return loss


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.9,
                 gamma: int = 2,
                 reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss = sigmoid_focal_loss(inputs=inputs,
                                  targets=targets,
                                  alpha=self.alpha,
                                  gamma=self.gamma,
                                  reduction=self.reduction)
        return loss


class DiceFocalLoss(nn.Module):
    def __init__(self, dice_kwargs={}, focal_kwargs={}):
        super().__init__()
        self.dice = DiceLoss(**dice_kwargs)
        self.focal = FocalLoss(**focal_kwargs)
    
    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        return dice_loss + focal_loss