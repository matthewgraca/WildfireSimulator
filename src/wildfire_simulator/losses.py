import torch
import torch.nn as nn


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
        self.last_mask_loss = mask_loss.item()
        self.last_arrival_loss = arrival_loss.item()
        return self.mask_weight * mask_loss + self.arrival_weight * arrival_loss
