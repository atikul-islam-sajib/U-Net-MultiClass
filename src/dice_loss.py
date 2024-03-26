import sys
import logging
import argparse
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filename="./logs/dice_loss.log",
    filemode="w",
)


class DiceLoss(nn.Module):
    def __init__(self, smooth=0.01):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, actual):
        predicted = predicted.view(-1)
        actual = actual.view(-1)

        intersection = (predicted * actual).sum()

        return (2 * intersection + self.smooth) / (
            predicted.sum() + actual.sum() + self.smooth
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dice Loss Example".title())
    parser.add_argument("--smooth", type=float, default=0.01)

    args = parser.parse_args()

    if args.smooth:
        dice_loss = DiceLoss(smooth=0.01)

        predicted = torch.tensor([0.2, 0.3, 0.5])
        actuals = torch.tensor([0, 1, 1])

        loss = dice_loss(predicted, actuals)
        logging.info(loss)
    else:
        raise Exception("Arguments should be defined properly".capitalize())
