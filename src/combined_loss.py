import sys
import logging
import argparse
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filename="./logs/combo_loss.log",
    filemode="w",
)


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.5, smooth=1e-6):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, predicted, target):
        predicted = predicted.view(-1)
        target = target.view(-1)

        BCELoss = nn.BCELoss()(predicted, target)

        DiceLoss = (2.0 * (predicted * target).sum() + self.smooth) / (
            predicted.sum() + target.sum() + self.smooth
        )

        pt = torch.exp(-BCELoss)

        FocalLoss = self.alpha * (1 - pt) ** self.gamma * BCELoss

        return (BCELoss + DiceLoss) / 2 + FocalLoss
