import sys
import logging
import argparse
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    filename="./logs/IoU.log",
    filemode="w",
)


class IoU(nn.Module):
    """
    A class for computing the Intersection over Union (IoU) metric, which is a common evaluation metric for models in tasks such as segmentation. The IoU is calculated by dividing the intersection of the predicted and actual areas by their union. This implementation includes a smoothing factor to avoid division by zero.

    | Attribute  | Type    | Description                                         |
    |------------|---------|-----------------------------------------------------|
    | smooth     | float   | A small constant added to avoid division by zero.   |
    | IoU        | float   | The last computed IoU value. `None` if not computed yet. |

    ## Methods

    - `__init__(self, smooth=1e-6)`
        Initializes the IoU metric object with a smoothing factor.

    - `forward(self, predicted, actual)`
        Computes the IoU metric for a batch of predictions and actual values.

    ## Examples

    ```python
    iou = IoU(smooth=1e-6)
    predicted = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=torch.float32)
    actual = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)
    loss = iou(predicted, actual)
    print(f"IoU score: {1 - loss.item()}")
    ```
    """

    def __init__(self, smooth=1e-6):
        """
        Initializes the IoU object with a smoothing factor to avoid division by zero in the IoU computation.

        | Parameter | Type  | Default | Description                                |
        |-----------|-------|---------|--------------------------------------------|
        | smooth    | float | 1e-6    | A small constant added for numerical stability. |
        """
        super(IoU, self).__init__()
        self.smooth = smooth
        self.IoU = None

    def forward(self, predicted, actual):
        """
        Computes the Intersection over Union (IoU) metric.

        | Parameters | Type          | Description                                   |
        |------------|---------------|-----------------------------------------------|
        | predicted  | torch.Tensor  | The predicted tensor. Should be a binary or probability map of shape (N, *) where N is the batch size. |
        | actual     | torch.Tensor  | The actual tensor. Should have the same shape as the predicted tensor. |

        | Returns    | Type          | Description                                   |
        |------------|---------------|-----------------------------------------------|
        | IoU_loss   | torch.Tensor  | The loss value, which is 1 - IoU score.       |

        The method modifies the internal state by updating the `IoU` attribute with the latest IoU score calculated.
        """
        predicted = predicted.view(-1)
        actual = actual.view(-1)

        intersection = (predicted * actual).sum()
        self.IoU = (2 * intersection + self.smooth) / (
            predicted.sum() + actual.sum() - intersection + self.smooth
        )
        return 1 - self.IoU


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tversky Loss".title())
    parser.add_argument(
        "--smooth",
        type=float,
        default=1e-6,
        help="Smoothing factor for Tversky Loss".capitalize(),
    )

    args = parser.parse_args()

    if args.smooth:
        iou = IoU(smooth=args.smooth)

        predicted = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=torch.float32)
        actual = torch.tensor([[0, 1, 0, 1], [1, 1, 1, 0]], dtype=torch.float32)

        logging.info(iou(predicted, actual))

    else:
        raise Exception("Arguments should be in a proper format".capitalize())
