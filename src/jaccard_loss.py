import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filename="./logs/jaccard_loss.log",
    filemode="w",
)


class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, actual):
        predicted = predicted.view(-1)
        actual = actual.view(-1)

        TP = (predicted * actual).sum()
        FP = ((1 - predicted) * actual).sum()
        FN = (predicted * (1 - actual)).sum()

        return 1 - ((TP + self.smooth) / (TP + FP + FN + self.smooth))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jaccard Loss".title())
    parser.add_argument(
        "--smooth", type=float, default=1e-6, help="smoothing factor".capitalize()
    )
    args = parser.parse_args()

    if args.smooth:
        loss = JaccardLoss(smooth=args.smooth)

        predicted = torch.tensor([0.2, 0.3, 0.5])
        actual = torch.tensor([0.2, 0.3, 0.1])

        loss_value = loss(predicted, actual)

        print(f"Jaccard Loss: {loss_value.item()}")

    else:
        raise Exception("Arguments should be provided".capitalize())
