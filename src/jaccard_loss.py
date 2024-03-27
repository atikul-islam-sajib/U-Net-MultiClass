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
    """
    Implements the Jaccard Loss, also known as the Intersection over Union (IoU) loss. This loss function measures the similarity between the predicted and actual values, and is particularly useful for segmentation tasks. The Jaccard index is calculated as the size of the intersection divided by the size of the union of the two label sets.

    | Attribute | Type  | Description                                       |
    |-----------|-------|---------------------------------------------------|
    | smooth    | float | A small constant added to avoid division by zero. |

    ## Methods

    - `__init__(self, smooth=1e-6)`
        Initializes the JaccardLoss object with a smoothing factor.

    - `forward(self, predicted, actual)`
        Computes the Jaccard loss for the given predictions and actual values.

    ## Examples

    ```python
    jaccard_loss = JaccardLoss(smooth=1e-6)
    predicted = torch.sigmoid(torch.randn(10, 1, requires_grad=True))
    actual = torch.empty(10, 1).random_(2)
    loss = jaccard_loss(predicted, actual)
    print(f"Jaccard Loss: {loss.item()}")
    ```

    Note: The predicted values should be probabilities obtained after applying the sigmoid or softmax function, and actual values should be binary (0 or 1) for the calculation to be meaningful.
    """

    def __init__(self, smooth=1e-6):
        """
        Initializes the JaccardLoss function with a specified smoothing value to avoid division by zero.

        | Parameter | Type  | Default | Description                                 |
        |-----------|-------|---------|---------------------------------------------|
        | smooth    | float | 1e-6    | A small constant added for numerical stability. |
        """
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, actual):
        """
        Computes the Jaccard Loss between predicted and actual outcomes.

        | Parameters | Type          | Description                                                        |
        |------------|---------------|--------------------------------------------------------------------|
        | predicted  | torch.Tensor  | Predicted outputs by the model. Must be of shape (N, *) where N is the batch size. |
        | actual     | torch.Tensor  | Ground truth values. Must be the same shape as predicted.          |

        | Returns    | Type          | Description                                                        |
        |------------|---------------|--------------------------------------------------------------------|
        | loss       | torch.Tensor  | The calculated Jaccard loss.                                       |
        """
        predicted = predicted.view(-1)
        actual = actual.view(-1)

        true_positives = (predicted * actual).sum()
        false_positives = ((1 - predicted) * actual).sum()
        false_negatives = (predicted * (1 - actual)).sum()

        jaccard_index = (true_positives + self.smooth) / (
            true_positives + false_positives + false_negatives + self.smooth
        )
        return 1 - jaccard_index


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
