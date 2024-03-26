import sys
import os
import logging
import argparse
import imageio
import matplotlib.pyplot as plt
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/test.log",
)

sys.path.append("src/")

from config import (
    BEST_MODEL_PATH,
    PROCESSED_PATH,
    TEST_IMAGE_PATH,
    TRAIN_IMAGES_PATH,
    GIF_PATH,
)
from utils import define_device, ignore_warnings, load
from UNet import UNet


class Charts:
    def __init__(self, device="mps"):
        self.device = define_device(device=device)
        ignore_warnings()

    def select_best_model(self):
        if os.path.exists(BEST_MODEL_PATH):
            return torch.load(os.path.join(BEST_MODEL_PATH, "best_model.pth"))
        else:
            raise Exception("Best model not found.".capitalize())

    def obtain_dataloader(self):
        if os.path.exists(PROCESSED_PATH):
            return load(os.path.join(PROCESSED_PATH, "test_dataloader.pkl"))
        else:
            raise Exception(
                "Processed data not found. Please run the preprocessing notebook.".capitalize()
            )

    def data_normalized(self, **kwargs):
        return (kwargs["data"] - kwargs["data"].min()) / (
            kwargs["data"].max() - kwargs["data"].min()
        )

    def plot_data_comparison(self):
        plt.figure(figsize=(40, 35))

        images, original_masks = next(iter(self.obtain_dataloader()))
        images = images.to(self.device)
        original_masks = original_masks.to(self.device)
        predicted_masks = self.model(images)

        for index, image in enumerate(images):
            plt.subplot(3 * 6, 3 * 4, 3 * index + 1)
            image = image.permute(1, 2, 0).detach().cpu().numpy()
            image = self.data_normalized(data=image)
            plt.imshow(image)
            plt.title("Image")
            plt.axis("off")

            plt.subplot(3 * 6, 3 * 4, 3 * index + 2)
            mask = original_masks[index].permute(1, 2, 0).detach().cpu().numpy()
            mask = self.data_normalized(data=mask)
            plt.imshow(mask)
            plt.title("Masks")
            plt.axis("off")

            plt.subplot(3 * 6, 3 * 4, 3 * index + 3)
            pred_mask = predicted_masks[index].permute(1, 2, 0).detach().cpu().numpy()
            pred_mask = self.data_normalized(data=pred_mask)
            plt.imshow(pred_mask)
            plt.title("Predicted")
            plt.axis("off")

        try:
            if os.path.exists(TEST_IMAGE_PATH):
                plt.tight_layout()
                plt.savefig(os.path.join(TEST_IMAGE_PATH, "result.png"))
            else:
                os.makedirs(TEST_IMAGE_PATH)
        except Exception as e:
            print("The exception in the section # {}".format(e).capitalize())
        else:
            plt.show()

    def generate_gif(self):
        if os.path.exists(TRAIN_IMAGES_PATH) and os.path.exists(GIF_PATH):
            images = [
                imageio.imread(os.path.join(TRAIN_IMAGES_PATH, image))
                for image in os.listdir(TRAIN_IMAGES_PATH)
            ]
            imageio.mimsave(os.path.join(GIF_PATH, "train_masks.gif"), images, "GIF")
        else:
            raise Exception("Train images path not found.".capitalize())

    def test(self):
        self.model = UNet().to(self.device)
        try:
            self.model.load_state_dict(self.select_best_model())
        except Exception as e:
            print("The exception in the section # {}".format(e).capitalize())
        try:
            self.plot_data_comparison()
        except Exception as e:
            print("The exception in the section # {}".format(e).capitalize())
        try:
            self.generate_gif()
        except Exception as e:
            print("The exception in the section # {}".format(e).capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing the model.".title())
    parser.add_argument(
        "--device", type=str, default="mps", help="Device to run the model on."
    )

    args = parser.parse_args()

    if args.device:
        charts = Charts(device="mps")
        charts.test()
    else:
        raise Exception("Device not specified.".capitalize())
