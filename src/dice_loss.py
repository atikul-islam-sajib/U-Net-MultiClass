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
    """
    ## DiceLoss Class

    Implements the Dice Loss function, commonly used for comparing pixel-wise agreement between predicted segmentation maps and their corresponding ground truth, with smoothing to handle division by zero errors. Particularly useful in binary and multi-class segmentation tasks in medical image processing.

    ### Parameters

    | Parameter | Type  | Default | Description                                                                 |
    |-----------|-------|---------|-----------------------------------------------------------------------------|
    | `smooth`  | float | 0.01    | A small constant added to the numerator and denominator to avoid division by zero errors. |

    ### Attributes

    | Attribute | Type  | Description                                                 |
    |-----------|-------|-------------------------------------------------------------|
    | `smooth`  | float | Smoothing factor to avoid division by zero errors.          |

    ### Examples

    ```python
    import torch
    from your_module import DiceLoss  # Make sure to replace 'your_module' with the actual module name

    dice_loss = DiceLoss(smooth=0.01)
    predicted = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
    actual = torch.tensor([0, 1, 1], dtype=torch.float32)
    loss = dice_loss(predicted, actual)
    print(loss)
    ```
    """

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
