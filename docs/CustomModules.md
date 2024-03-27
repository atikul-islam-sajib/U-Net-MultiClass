### DataLoader

The `Loader` class is responsible for preparing the dataset. It unzips the dataset, splits it into training and testing sets based on the provided ratio, and creates DataLoaders for both.

To use the DataLoader, ensure you have your dataset in a zip file. Specify the path to this file along with other parameters such as batch size, split ratio, and image size.

Example:

```python
from src.dataloader import Loader

loader = Loader(
    image_path="path/to/your/dataset.zip",
    batch_size=4,
    split_ratio=0.25,
    image_size=128
)
loader.unzip_folder()
loader.create_dataloader()
```

### Loss Functions

The training process supports several loss functions, allowing you to choose the one that best fits your project's needs. Below is a table describing the available loss functions and how to specify each in the training command or configuration.

| Loss Function | Call              | Description                                                                                                                                                      |
| ------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dice          | `loss="dice"`     | Measures the overlap between the predicted segmentation and the ground truth. Ideal for binary segmentation tasks.                                               |
| Jaccard       | `loss="jaccard"`  | Also known as the Intersection over Union (IoU) loss. Similar to Dice but with a different formula. Good for evaluating the accuracy of object detection models. |
| IoU           | `loss="IoU"`      | Another name for Jaccard loss.                                                                                                                                   |
| Combo         | `loss="combo"`    | Combines Dice and a cross-entropy loss to leverage the benefits of both. Useful for unbalanced datasets.                                                         |
| Focal         | `loss="focal"`    | Focuses on hard-to-classify examples by reducing the relative loss for well-classified examples. Useful for datasets with imbalanced classes.                    |
| Dice_BCE      | `loss="dice_bce"` | A combination of Dice and Binary Cross-Entropy (BCE) losses. Offers a balance between shape similarity and pixel-wise accuracy.                                  |
| None          | `loss=None`       | IT will trigger the BCELoss                                                                                                                                      |

### Trainer

The `Trainer` class manages the training process, including setting up the loss function, optimizer, and device (CPU, CUDA, MPS). It also handles the training epochs and displays progress if enabled.

To train your model, configure the `Trainer` with your desired settings.

Example:

```python
from src.trainer import Trainer

trainer = Trainer(
    epochs=100,
    lr=0.01,
    loss="dice", # can be "jaccard", "IoU", "combo", "focal", "dice_bce", "None"
    alpha=0.5,
    gamma=2,
    display=True,
    device="cuda",  # Can be "cpu", "cuda", or "mps"
    smooth_value=0.01
)
trainer.train()
```

### Charts

After training, you can test and visualize the model's performance using the `Charts` class. This class allows you to evaluate the trained model on the test dataset and generate performance metrics.

Example:

```python
from src.test import Charts

charts = Charts(device="mps")  # Specify the device used for testing
charts.test()
```
