import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridLoss(nn.Module):
    """
    Combined loss for fire spread prediction:
    - BCE on channel 0 (fire mask — binary classification)
    - MSE on channel 1 (arrival time — continuous regression)
    """
    def __init__(self, mask_weight=1.0, arrival_weight=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.mask_weight = mask_weight
        self.arrival_weight = arrival_weight
        # Expose individual losses for logging
        self.last_mask_loss = 0.0
        self.last_arrival_loss = 0.0

    def forward(self, pred, target):
        mask_loss = self.bce(pred[:, 0:1], target[:, 0:1])
        arrival_loss = self.mse(pred[:, 1:2], target[:, 1:2])
        self.last_mask_loss = (self.mask_weight * mask_loss).item()
        self.last_arrival_loss = (self.arrival_weight * arrival_loss).item()
        return self.mask_weight * mask_loss + self.arrival_weight * arrival_loss


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class FocalLoss(nn.Module):
    """Focal loss for binary segmentation (handles class imbalance)."""
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class FireSenseNetLoss(nn.Module):
    """
    Combined loss from FireSenseNet (arxiv 2604.07675):
    - Weighted BCE (positive class weight)
    - Dice loss (spatial overlap)
    - Focal loss (focus on hard boundary pixels)

    L = bce_weight * BCE_w + dice_weight * Dice + focal_weight * Focal
    """
    def __init__(self, bce_weight=0.4, dice_weight=0.3, focal_weight=0.3,
                 pos_weight=3.0, focal_gamma=2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.pos_weight = pos_weight

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=focal_gamma)

        # Expose individual weighted losses for logging
        self.last_bce = 0.0
        self.last_dice = 0.0
        self.last_focal = 0.0

    def forward(self, pred, target):
        # Weighted BCE: fire pixels weighted by pos_weight
        weight = torch.where(target == 1, self.pos_weight, 1.0)
        bce = F.binary_cross_entropy(pred, target, weight=weight, reduction='mean')

        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)

        # Log weighted contributions
        self.last_bce = (self.bce_weight * bce).item()
        self.last_dice = (self.dice_weight * dice).item()
        self.last_focal = (self.focal_weight * focal).item()

        return self.bce_weight * bce + self.dice_weight * dice + self.focal_weight * focal
