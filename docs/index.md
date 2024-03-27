# U-Net for Semantic Image Segmentation

<img src="https://ci3.googleusercontent.com/mail-img-att/AGAZnRoKAF94mRUqQBcmh62EXZXV0EHo4NeBuP9L48viCYMtYXZ3tt_LorYWBbKewNN3w2UXsJSceaaNpmyRtqRYp34aH-7eqt98SM_g5Ai5BEH87-S_NMYJnlI28xPiKiDmd1gLb4pDZA_CQPo4zofQd8fQ292DDgKkJLZwS7uguse_PSrMBLEz5Qt3ZS-MpxpMgncWB__8O16-ISHvfxe4SvlbNYwMSmuhHeLkkBVRTu5j2sSIc780sMdQ3wSamj9EbvzNeeBWM_O0HkdUAYrUTgcjG5RLoJb8HYu4ViOrqgODJJ5SDnGlJC58a3Pwi175YN76KqEcBWfM3qoYAmeefe62UYxwSUaaLtl7px42PRqAdtz3LqZHhAXp-DZ7NW_AwpZzt1k7RgreB1ErRCd7-xULPPtpypO1JqKZ8YjjOwI_gquhmi_xRUOyOg_1-bMVOunD7IE9uZCnaGK0nqHQSBLrKiFoQHrKxhK4kSNVK1bGXK5PlGAhrxG4GDus7JIYYBq03fJPhiQ1KyKVR5Q5mRqjXa47dI2WeXir8jkN_z-BS6ePwC3pJrOPZbCpVGoLS1ygtjvyZj-4jRQR35v_ZRWcuyWegdM4L1cQ2iDz22smT8-aUnbxUFEH25FTUZFvXVhCi0mTUYEVLlOSJ6Z3GaPzzNmyREJDph55PTOu3vfDd6I_r02nzX_gJTNhVr1zNltXND7DES3d-VOnI9Du1n8mjPtvMwkiAsWpmLqUt1o_yOUgi5CMJN8-h59GNMMaJWbDayNPSTOisgnwWwW_e9XK--fao-lQaVGEeKIhmrPzgPAKZnv0UzgmM2l6n-0SKxLNRwhKxFPzCpzyQmJKnrhAVNpWMV8x3j8sKf2bi3QaSYbAYpWVDVU1-N7KjQO9piEspDlH-sP_L0xZYxXGarEdD3E4n-F10kLfOB5FJt-Raz5hV7WHyK2w-2lK0rhnmazzASw-DGsDvnRUflOzWbfiLsdx-C5Q2AP6KL5cCam8EmaNQyn9Ar9v_HUphZVXoHcXJb7gJTiHukMU=s0-l75-ft" alt="AC-GAN - Medical Image Dataset Generator: Generated Image with labels">

U-Net is a convolutional neural network designed for semantic image segmentation. This implementation of U-Net is tailored for high performance on various image segmentation tasks, allowing for precise object localization within images.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20220614121231/Group14.jpg" alt="AC-GAN - Medical Image Dataset Generator: Generated Image with labels">

## Features

| Feature                          | Description                                                                                                                                                                                                           |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Efficient Implementation**     | Utilizes an optimized U-Net model architecture for superior performance on diverse image segmentation tasks.                                                                                                          |
| **Custom Dataset Support**       | Features easy-to-use data loading utilities that seamlessly accommodate custom datasets, requiring minimal configuration.                                                                                             |
| **Training and Testing Scripts** | Provides streamlined scripts for both training and testing phases, simplifying the end-to-end workflow.                                                                                                               |
| **Visualization Tools**          | Equipped with tools for tracking training progress and visualizing segmentation outcomes, enabling clear insight into model effectiveness.                                                                            |
| **Custom Training via CLI**      | Offers a versatile command-line interface for personalized training configurations, enhancing flexibility in model training.                                                                                          |
| **Import Modules**               | Supports straightforward integration into various projects or workflows with well-documented Python modules, simplifying the adoption of U-Net functionality.                                                         |
| **Multi-Platform Support**       | Guarantees compatibility with various computational backends, including MPS for GPU acceleration on Apple devices, CPU, and CUDA for Nvidia GPU acceleration, ensuring adaptability across different hardware setups. |

## Demo - During training

![AC-GAN - Medical Image Dataset Generator with class labels: Gif file](https://raw.githubusercontent.com/atikul-islam-sajib/U-Net-MultiClass/main/outputs/train_gif/train_masks.gif)

## Getting Started

## Requirements

| Requirement             | Description                                                                                                     |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Python Version**      | Python 3.9 or newer is required for compatibility with the latest features and library support.                 |
| **CUDA-compatible GPU** | Access to a CUDA-compatible GPU is recommended for training and testing with CUDA acceleration.                 |
| **Python Libraries**    | Essential libraries include: **torch**, **matplotlib**, **numpy**, **PIL**, **scikit-learn**, **opencv-python** |

## Installation Instructions

Follow these steps to get the project set up on your local machine:

| Step | Instruction                                  | Command                                                                  |
| ---- | -------------------------------------------- | ------------------------------------------------------------------------ |
| 1    | Clone this repository to your local machine. | **git clone https://github.com/atikul-islam-sajib/U-Net-MultiClass.git** |
| 2    | Navigate into the project directory.         | **cd U-Net-MultiClass**                                                  |
| 3    | Install the required Python packages.        | **pip install -r requirements.txt**                                      |

## Project Structure

This project is thoughtfully organized to support the development, training, and evaluation of the U-Net model efficiently. Below is a concise overview of the directory structure and their specific roles:

- **checkpoints/**
  - Stores model checkpoints during training for later resumption.
- **best_model/**

  - Contains the best-performing model checkpoints as determined by validation metrics.

- **train_models/**

  - Houses all model checkpoints generated throughout the training process.

- **data/**

  - **processed/**: Processed data ready for modeling, having undergone normalization, augmentation, or encoding.
  - **raw/**: Original, unmodified data serving as the baseline for all preprocessing.

- **logs/**

  - **Log** files for debugging and tracking model training progress.

- **metrics/**

  - Files related to model performance metrics for evaluation purposes.

- **outputs/**

  - **test_images/**: Images generated during the testing phase, including segmentation outputs.
  - **train_gif/**: GIFs compiled from training images showcasing the model's learning progress.
  - **train_images/**: Images generated during training for performance visualization.

- **research/**

  - **notebooks/**: Jupyter notebooks for research, experiments, and exploratory analyses conducted during the project.

- **src/**

  - Source code directory containing all custom modules, scripts, and utility functions for the U-Net model.

- **unittest/**
  - Unit tests ensuring code reliability, correctness, and functionality across various project components.

### Dataset Organization for Semantic Image Segmentation

The dataset is organized into three categories for semantic image segmentation tasks: benign, normal, and malignant. Each category directly contains paired images and their corresponding segmentation masks, stored together to simplify the association between images and masks.

## Directory Structure:

```
segmentation/
├── Original/
│ ├── Benign/
│ │ ├── benign(1).png
│ │ ├── benign(2).png
│ │ ├── ...
│ ├── Early/
│ │ ├── early(1).png
│ │ ├── early(2).png
│ │ ├── ...
│ ├── Pre/
│ │ ├── pre(1).png
│ │ ├── pre(2).png
│ │ ├── ...
│ ├── Pro/
│ │ ├── pro(1).png
│ │ ├── pro(2).png
│ │ ├── ...
├── Segmented/
│ ├── Benign/
│ │ ├── benign(1).png
│ │ ├── benign(2).png
│ │ ├── ...
│ ├── Early/
│ │ ├── early(1).png
│ │ ├── early(2).png
│ │ ├── ...
│ ├── Pre/
│ │ ├── pre(1).png
│ │ ├── pre(2).png
│ │ ├── ...
│ ├── Pro/
│ │ ├── pro(1).png
│ │ ├── pro(2).png
│ │ ├── ...

```

#### Naming Convention:

- **Images and Masks**: Within each category folder, images and their corresponding masks are stored together. The naming convention for images is `[category](n).png`, and for masks, it is in Segmented `[category](n).png`, where `[category]` represents the type of the image (benign, normal, or malignant), and `(n)` is a unique identifier. This convention facilitates easy identification and association of each image with its respective mask.

For detailed documentation on the dataset visit the [Dataset - Kaggle](https://www.kaggle.com/datasets/mehradaria/leukemia).

## Detailed Documentation Links

| Component      | Documentation Link                   |
| -------------- | ------------------------------------ |
| DataLoader     | [DataLoader](./dataloader.md)        |
| Encoder        | [Encoder](./encoder.md)              |
| Decoder        | [Decoder](./decoder.md)              |
| U-Net          | [U-Net](./UNet.md)                   |
| DiceLoss       | [DiceLoss](./DiceLoss.md)            |
| FocalLoss      | [FocalLoss](./FocalLoss.md)          |
| JaccardLoss    | [JaccardLoss](./JaccardLoss.md)      |
| TverskyLoss    | [TverskyLoss](./TverskyLoss.md)      |
| DiceBCELoss    | [DiceBCELoss](./DiceBCELoss.md)      |
| ComboLoss      | [ComboLoss](./ComboLoss.md)          |
| Trainer        | [Trainer](./trainer.md)              |
| Charts         | [Charts](./Charts.md)                |
| CLI            | [CLI](./CLI.md)                      |
| Custom Modules | [Custom Modules](./CustomModules.md) |

## Contributing

Contributions to improve this implementation of U-Net are welcome. Please follow the standard fork-branch-pull request workflow.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
