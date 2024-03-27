# UNet Training and Testing CLI Tool

This CLI tool is designed for training and testing the UNet model on image segmentation tasks. It supports various configurations through command-line arguments, allowing users to customize the training and testing processes.

## Usage

```
python cli.py [--image_path PATH] [--batch_size SIZE] [--split_ratio RATIO] [--image_size SIZE] [--epochs EPOCHS] [--lr LEARNING_RATE] [--loss LOSS] [--display DISPLAY] [--device DEVICE] [--smooth_value VALUE] [--alpha ALPHA] [--gamma GAMMA] [--train] [--test]
```

### Arguments

| Argument         | Type    | Default | Description                                                        |
| ---------------- | ------- | ------- | ------------------------------------------------------------------ |
| `--image_path`   | `str`   | None    | Path to the zip file containing the images.                        |
| `--batch_size`   | `int`   | 4       | Batch size for the DataLoader.                                     |
| `--split_ratio`  | `float` | 0.25    | Ratio to split the dataset into train and test sets.               |
| `--image_size`   | `int`   | 128     | Size of the images.                                                |
| `--epochs`       | `int`   | 100     | Number of epochs to train.                                         |
| `--lr`           | `float` | 1e-2    | Learning rate.                                                     |
| `--loss`         | `str`   | "dice"  | Loss function to use.                                              |
| `--display`      | `bool`  | True    | Whether to display progress during training.                       |
| `--device`       | `str`   | "mps"   | Device to use for training/testing. Options: "cpu", "cuda", "mps". |
| `--smooth_value` | `float` | 0.01    | Smooth value for loss calculation.                                 |
| `--alpha`        | `float` | 0.5     | Alpha value for loss calculation.                                  |
| `--gamma`        | `float` | 2       | Gamma value for loss calculation.                                  |
| `--train`        | Action  | False   | Flag to initiate training process.                                 |
| `--test`         | Action  | False   | Flag to initiate testing process.                                  |

### Supported Loss Functions

The CLI tool supports various loss functions, each with specific parameters for fine-tuning the training process.

| Loss Function | Parameters                             | Description                                                                              |
| ------------- | -------------------------------------- | ---------------------------------------------------------------------------------------- |
| `DiceLoss`    | `smooth=0.01`                          | Dice loss function with a smooth value to avoid division by zero.                        |
| `DiceBCELoss` | `smooth=0.01`                          | Combination of Dice and Binary Cross-Entropy (BCE) losses.                               |
| `FocalLoss`   | `alpha=0.25`, `gamma=2`                | Focal loss function, useful for unbalanced classes. Alpha and Gamma are hyperparameters. |
| `TverskyLoss` | `smooth=0.01`                          | Tversky loss function, a generalization of Dice loss.                                    |
| `JaccardLoss` | `smooth=0.01`                          | Jaccard loss function, a generalization of Dice loss.                                    |
| `ComboLoss`   | `smooth=0.01`, `alpha=0.25`, `gamma=2` | Combo loss function, a generalization of Dice loss, FocalLoss and BCELoss                |

### Training and Testing

#### Training the Model

To train the model, you need a dataset in a zip file specified by `--image_path`, along with any other configurations you wish to customize.

- **Using CUDA (for NVIDIA GPUs):**

```
python cli.py --image_path "/path/to/dataset.zip" --batch_size 4 --image_size 128 --split_ratio 0.25 --epochs 50 --lr 0.001 --loss dice --display True --smooth_value 0.01 --alpha 0.25 --gamma 2 --device cuda --train
```

- **Using MPS (for Apple Silicon GPUs):**

```
python cli.py --image_path "/path/to/dataset.zip" --batch_size 4 --image_size 128 --split_ratio 0.25 --epochs 50 --lr 0.001 --loss dice --display True --smooth_value 0.01 --alpha 0.25 --gamma 2  --device mps --train
```

- **Using CPU:**

```
python cli.py --image_path "/path/to/dataset.zip" --batch_size 4 --image_size 128 --split_ratio 0.25 --epochs 50 --lr 0.001 --loss dice --display True --smooth_value 0.01 --alpha 0.25 --gamma 2 --device cpu --train
```

#### Testing the Model

Ensure you specify the device using `--device` if different from the default. The test process can be initiated with the `--test` flag.

- **Using CUDA (for NVIDIA GPUs):**

```
python cli.py --device cuda --test
```

- **Using MPS (for Apple Silicon GPUs):**

```
python cli.py --device mps --test
```

- **Using CPU:**

```
python cli.py --device cpu --test
```
