import sys
import logging
import argparse
import os
import zipfile
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    filemode="w",
    filename="./logs/dataloader.log",
)

sys.path.append("src/")

from config import RAW_PATH, PROCESSED_PATH
from utils import dump, load


class Loader:
    """
    A class for loading, processing, and creating DataLoader objects for segmented image datasets.

    | Attributes      | Description                                           |
    |-----------------|-------------------------------------------------------|
    | image_path      | Path to the zip file containing the dataset.         |
    | batch_size      | The size of batches to use when creating DataLoaders. |
    | base_images     | List to store base images after processing.           |
    | mask_images     | List to store mask images after processing.           |

    | Parameters      | Type  | Description                                      |
    |-----------------|-------|--------------------------------------------------|
    | image_path      | str   | Path to the zip file containing the dataset.     |
    | batch_size      | int   | The size of batches to use when creating DataLoaders. |


    ## Usage:

    ### From the Command Line:

    To use this script from the command line, navigate to the directory containing the script and execute it with the required arguments. For example:

    ```
    python loader_script.py --image_path /path/to/your/dataset.zip --batch_size 4
    ```

    ### From the Modules:
    ```
    # Initialize the Loader with the path to your dataset zip file and the desired batch size
    loader = Loader(image_path='/path/to/your/dataset.zip', batch_size=4)

    # Unzip the dataset
    loader.unzip_folder()

    # Process the segmented data and create DataLoader objects
    loader.create_dataloader()

    # Optionally, you can display images from the validation set
    loader.show_images()
    ```

    """

    def __init__(self, image_path=None, batch_size=4):
        self.image_path = image_path
        self.batch_size = batch_size
        self.base_images = list()
        self.mask_images = list()

    def base_transforms(self):
        """
        Defines and returns a torchvision transforms pipeline for processing base images.

        | Returns         | Description                                     |
        |-----------------|-------------------------------------------------|
        | transforms.Compose | A composition of image transformations for base images. |
        """
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def mask_transforms(self):
        """
        Defines and returns a torchvision transforms pipeline for processing mask images.

        | Returns         | Description                                      |
        |-----------------|--------------------------------------------------|
        | transforms.Compose | A composition of image transformations for mask images. |
        """
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def unzip_folder(self):
        """
        Extracts the dataset from a zip file into the RAW_PATH directory.

        | Exceptions      | Description                                      |
        |-----------------|--------------------------------------------------|
        | Exception       | Raised if the RAW_PATH directory does not exist. |
        """
        if os.path.exists(RAW_PATH):
            with zipfile.ZipFile(self.image_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(RAW_PATH, "segmented"))
        else:
            raise Exception("Raw data folder not found".capitalize())

    def process_segmented_data(self):
        """
        Processes segmented images and masks, loading them into memory after applying necessary transformations.

        | Returns         | Description                                      |
        |-----------------|--------------------------------------------------|
        | tuple           | A tuple containing lists of processed base images and masks. |

        | Exceptions      | Description                                                  |
        |-----------------|--------------------------------------------------------------|
        | Exception       | Raised if the segmented data folder is not found in RAW_PATH. |
        """
        if os.path.join(RAW_PATH, "segmented"):
            self.images_directory = os.path.join(RAW_PATH, "segmented")

            self.base_directory = os.path.join(
                self.images_directory, os.listdir(self.images_directory)[0]
            )
            self.mask_directory = os.path.join(
                self.images_directory, os.listdir(self.images_directory)[1]
            )

            self.categories = os.listdir(self.base_directory)

            for category in self.categories:
                base_folder_path = os.path.join(self.base_directory, category)
                mask_folder_path = os.path.join(self.mask_directory, category)

                for image in os.listdir(base_folder_path):
                    if image in os.listdir(mask_folder_path):
                        self.base_images.append(
                            self.base_transforms()(
                                Image.fromarray(
                                    cv2.imread(os.path.join(base_folder_path, image))
                                )
                            )
                        )
                        self.mask_images.append(
                            self.mask_transforms()(
                                Image.fromarray(
                                    cv2.imread(os.path.join(mask_folder_path, image))
                                )
                            )
                        )

            return self.base_images, self.mask_images
        else:
            raise Exception(
                "Segmented data folder not found in the raw folder".capitalize()
            )

    def create_dataloader(self):
        """
        Creates training and validation DataLoader objects from processed images and masks.

        | Exceptions      | Description                                         |
        |-----------------|-----------------------------------------------------|
        | Exception       | Raised if the PROCESSED_PATH directory does not exist or if any error occurs during DataLoader creation. |
        """
        images, masks = self.process_segmented_data()
        data_split = train_test_split(images, masks, test_size=0.30, random_state=42)

        train_dataloader = DataLoader(
            dataset=list(zip(data_split[0], data_split[2])),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            dataset=list(zip(data_split[1], data_split[3])),
            batch_size=self.batch_size * 6,
            shuffle=True,
        )

        try:
            if os.path.exists(PROCESSED_PATH):
                dump(
                    value=train_dataloader,
                    filename=os.path.join(PROCESSED_PATH, "train_dataloader.pkl"),
                )
                dump(
                    value=val_dataloader,
                    filename=os.path.join(PROCESSED_PATH, "val_dataloader.pkl"),
                )
            else:
                raise Exception("Processed data folder not found".capitalize())

        except ValueError as e:
            print("Exception caught in the section - {}".format(e).capitalize())

    @staticmethod
    def show_images():
        """
        Displays a set of images and masks from the validation DataLoader.

        | Exceptions      | Description                                         |
        |-----------------|-----------------------------------------------------|
        | Exception       | Raised if the PROCESSED_PATH directory does not exist. |
        """
        plt.figure(figsize=(30, 20))

        if os.path.exists(PROCESSED_PATH):
            val_images, val_masks = next(
                iter(load(os.path.join(PROCESSED_PATH, "val_dataloader.pkl")))
            )
            for index, image in enumerate(val_images):
                plt.subplot(2 * 4, 2 * 6, 2 * index + 1)
                image = image.permute(1, 2, 0)
                image = (image - image.min()) / (image.max() - image.min())
                plt.imshow(image)
                plt.title("Image")
                plt.axis("off")

                plt.subplot(2 * 4, 2 * 6, 2 * index + 2)
                masks = val_masks[index].permute(1, 2, 0)
                masks = (masks - masks.min()) / (masks.max() - masks.min())
                plt.imshow(masks, cmap="gray")
                plt.title("Mask")
                plt.axis("off")

            plt.tight_layout()
            plt.show()

        else:
            raise Exception("Processed data folder not found".capitalize())


if __name__ == "__main__":
    """
    Entry point for the script. Parses command-line arguments to initialize a Loader instance and process the dataset.

    | Parameters      | Type  | Description                                      |
    |-----------------|-------|--------------------------------------------------|
    | --image_path    | str   | Path to the zip file containing the dataset.     |
    | --batch_size    | int   | The size of batches to use when creating DataLoaders. |

    To use this script from the command line, navigate to the directory containing the script and execute it with the required arguments. For example:

    ```
    python loader_script.py --image_path /path/to/your/dataset.zip --batch_size 4
    ```
    """
    parser = argparse.ArgumentParser(description="Data Loader for U-Net".title())
    parser.add_argument(
        "--image_path", type=str, help="Path to the zip file".capitalize()
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for the dataloader".capitalize(),
    )
    args = parser.parse_args()

    if args.image_path and args.batch_size:
        logging.info("Data Loader started".capitalize())

        loader = Loader(image_path=args.image_path, batch_size=args.batch_size)
        loader.unzip_folder()
        loader.create_dataloader()

        logging.info("Data Loader completed".capitalize())

        loader.show_images()
    else:
        raise Exception("Invalid arguments".capitalize())
