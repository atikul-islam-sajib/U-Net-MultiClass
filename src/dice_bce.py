import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filename="./logs/dice_bce.log",
    filemode="w",
)


class DiceBCE(nn.Module):
    """
    Implements a combination of Dice Loss and Binary Cross-Entropy (BCE) Loss. This hybrid loss function is useful for segmentation tasks, combining the benefits of Dice Loss for handling class imbalance with the probabilistic classification capabilities of BCE Loss.

    | Attribute | Type  | Description                                   |
    |-----------|-------|-----------------------------------------------|
    | smooth    | float | Smoothing factor to avoid division by zero.  |

    ## Methods

    - `__init__(self, smooth=0.01)`
        Initializes the DiceBCE object with a smoothing factor.

    - `forward(self, predicted, actual)`
        Computes the combined Dice and BCE loss for the given predictions and actual values.

    ## Examples

    ```python
    dice_bce_loss = DiceBCE(smooth=0.01)
    predicted = torch.sigmoid(torch.randn(10, 1, requires_grad=True))
    actual = torch.empty(10, 1).random_(2)
    loss = dice_bce_loss(predicted, actual)
    print(f"Combined Dice-BCE Loss: {loss.item()}")
    ```

    Note: The predicted values should be probabilities obtained after applying the sigmoid or softmax function, and actual values should be binary (0 or 1) for the calculation to be meaningful.
    """

    def __init__(self, smooth=0.01):
        """
        Initializes the DiceBCE function with a specified smoothing value to avoid division by zero.

        | Parameter | Type  | Default | Description                                 |
        |-----------|-------|---------|---------------------------------------------|
        | smooth    | float | 0.01    | A small constant added for numerical stability. |
        """
        super(DiceBCE, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, actual):
        """
        Computes the combined Dice and BCE loss.

        | Parameters | Type          | Description                                                        |
        |------------|---------------|--------------------------------------------------------------------|
        | predicted  | torch.Tensor  | Predicted outputs by the model. Must be of shape (N, *) where N is the batch size. |
        | actual     | torch.Tensor  | Ground truth values. Must be the same shape as predicted.          |

        | Returns    | Type          | Description                                                        |
        |------------|---------------|--------------------------------------------------------------------|
        | combined_loss | torch.Tensor | The calculated combined Dice and BCE loss.                        |
        """
        predicted = predicted.contiguous().view(-1)
        actual = actual.contiguous().view(-1)

        intersection = (predicted * actual).sum()
        BCE = F.binary_cross_entropy(predicted, actual, reduction="mean")

        dice = 1 - (2 * intersection + self.smooth) / (
            predicted.sum() + actual.sum() + self.smooth
        )

        return dice + BCE
