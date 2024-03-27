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
    """
    Implements the Focal Loss function, which is designed to address class imbalance by decreasing the relative loss for well-classified examples, putting more focus on hard, misclassified examples. It is particularly useful in object detection where the imbalance between the background and foreground classes is significant.

    | Attribute | Type  | Description                                             |
    |-----------|-------|---------------------------------------------------------|
    | alpha     | float | The weighting factor for the class with fewer samples.  |
    | gamma     | int   | The focusing parameter to adjust the rate at which easy examples are down-weighted. |
    | smooth     | float   | A small constant added to the numerator and denominator to avoid division by zero errors |

    ## Methods

    - `__init__(self, alpha=0.25, gamma=2, smooth = 0.01)`
        Initializes the FocalLoss object with `alpha` and `gamma` and `smooth` parameters.

    - `forward(self, predicted, actual)`
        Computes the focal loss between `predicted` and `actual` values.

    ## Examples

    ```python
    combo_loss = CombolLoss(alpha=0.25, gamma=2, smooth = 0.01)
    predicted = torch.sigmoid(torch.randn(10, 1, requires_grad=True))
    actual = torch.empty(10, 1).random_(2)
    loss = combo_loss(predicted, actual)
    print(f"Loss: {loss.item()}")
    ```

    Note: The predicted values should be sigmoid probabilities, and actual values should be binary (0 or 1).
    """

    def __init__(self, alpha=0.5, gamma=0.5, smooth=1e-6):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, predicted, target):
        """
        Computes the Combo Loss between predicted and actual outcomes.

        | Parameters | Type          | Description                                                        |
        |------------|---------------|--------------------------------------------------------------------|
        | predicted  | torch.Tensor  | Predicted outputs (probabilities) by the model. Must be of shape (N, *) where N is the batch size. |
        | actual     | torch.Tensor  | Ground truth values. Must be the same shape as predicted.          |

        | Returns    | Type          | Description                                                        |
        |------------|---------------|--------------------------------------------------------------------|
        | loss       | torch.Tensor  | Computed Combo Loss.                                               |
        """
        predicted = predicted.view(-1)
        target = target.view(-1)

        BCELoss = nn.BCELoss()(predicted, target)

        DiceLoss = (2.0 * (predicted * target).sum() + self.smooth) / (
            predicted.sum() + target.sum() + self.smooth
        )

        pt = torch.exp(-BCELoss)

        FocalLoss = self.alpha * (1 - pt) ** self.gamma * BCELoss

        return (BCELoss + DiceLoss) / 2 + FocalLoss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Define the alpha parameter".capitalize(),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Define the gamma parameter".capitalize(),
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=1e-6,
        help="Define the smooth parameter".capitalize(),
    )

    args = parser.parse_args()

    if args.alpha and args.gamma and args.smooth:
        combo = ComboLoss(args.alpha, args.gamma, args.smooth)

        predicted = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=torch.float32)
        actual = torch.tensor([[0, 1, 0, 1], [1, 1, 1, 0]], dtype=torch.float32)

        print(combo(predicted, actual))

    else:
        raise Exception("Arguments not provided".capitalize())
