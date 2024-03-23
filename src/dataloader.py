import sys
import logging
import argparse
import os
import zipfile
import cv2
from PIL import Image
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
from utils import dump


class Loader:
    def __init__(self, image_path=None, batch_size=4):
        self.image_path = image_path
        self.batch_size = batch_size
        self.base_images = list()
        self.mask_images = list()

    def base_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def mask_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def unzip_folder(self):
        if os.path.exists(RAW_PATH):
            with zipfile.ZipFile(self.image_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(RAW_PATH, "segmented"))
        else:
            raise Exception("Raw data folder not found".capitalize())

    def process_segmented_data(self):
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


if __name__ == "__main__":
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
        loader = Loader(image_path=args.image_path, batch_size=args.batch_size)
        loader.unzip_folder()
        loader.create_dataloader()
    else:
        raise Exception("Invalid arguments".capitalize())
