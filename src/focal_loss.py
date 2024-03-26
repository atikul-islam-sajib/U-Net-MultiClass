import sys
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filename="./logs/focal_loss.log",
    filemode="w",
)


class FocalLoss(nn.Module):
    """
    Implements the Focal Loss function, which is designed to address class imbalance by decreasing the relative loss for well-classified examples, putting more focus on hard, misclassified examples. It is particularly useful in object detection where the imbalance between the background and foreground classes is significant.

    | Attribute | Type  | Description                                             |
    |-----------|-------|---------------------------------------------------------|
    | alpha     | float | The weighting factor for the class with fewer samples.  |
    | gamma     | int   | The focusing parameter to adjust the rate at which easy examples are down-weighted. |

    ## Methods

    - `__init__(self, alpha=0.25, gamma=2)`
        Initializes the FocalLoss object with `alpha` and `gamma` parameters.

    - `forward(self, predicted, actual)`
        Computes the focal loss between `predicted` and `actual` values.

    ## Examples

    ```python
    focal_loss = FocalLoss(alpha=0.25, gamma=2)
    predicted = torch.sigmoid(torch.randn(10, 1, requires_grad=True))
    actual = torch.empty(10, 1).random_(2)
    loss = focal_loss(predicted, actual)
    print(f"Loss: {loss.item()}")
    ```

    Note: The predicted values should be sigmoid probabilities, and actual values should be binary (0 or 1).
    """

    def __init__(self, alpha=0.25, gamma=2):
        """
        Initializes the FocalLoss function with specified alpha and gamma values.

        | Parameter | Type  | Default | Description                                                        |
        |-----------|-------|---------|--------------------------------------------------------------------|
        | alpha     | float | 0.25    | The alpha weighting factor to balance class weights.               |
        | gamma     | int   | 2       | The gamma focusing parameter to reduce the loss for easy examples. |
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predicted, actual):
        """
        Computes the Focal Loss between predicted and actual outcomes.

        | Parameters | Type          | Description                                                        |
        |------------|---------------|--------------------------------------------------------------------|
        | predicted  | torch.Tensor  | Predicted outputs (probabilities) by the model. Must be of shape (N, *) where N is the batch size. |
        | actual     | torch.Tensor  | Ground truth values. Must be the same shape as predicted.          |

        | Returns    | Type          | Description                                                        |
        |------------|---------------|--------------------------------------------------------------------|
        | loss       | torch.Tensor  | Computed Focal Loss.                                               |
        """
        predicted = predicted.view(-1)
        actual = actual.view(-1)

        BCE = F.binary_cross_entropy(predicted, actual)
        pt = torch.exp(-BCE)

        return self.alpha * (1 - pt) ** self.gamma * BCE
