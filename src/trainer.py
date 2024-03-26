import sys
import os
import logging
import argparse
import numpy as np
import torch
import torch.optim as optim

sys.path.append("src/")

from config import (
    PROCESSED_PATH,
    TRAIN_CHECKPOINT_PATH,
    TEST_CHECKPOINT_PATH,
    TRAIN_IMAGES_PATH,
)
from utils import load, dump, weight_init, define_device
from UNet import UNet


class Trainer:
    def __init__(
        self,
        epochs=10,
        lr=0.0002,
        loss="dice",
        smooth_value=0.01,
        beta1=0.5,
        device="mps",
        display=True,
    ):
        self.epochs = epochs
        self.lr = lr
        self.loss = loss
        self.smooth_value = smooth_value
        self.beta1 = beta1
        self.beta2 = 0.999
        self.weight_decay = 1e-4
        self.device = device
        self.is_display = display
        self.history = {"train_loss": list(), "test_loss": list()}

    def __setup__(self):
        self.device = define_device(device=self.device)
        self.model = UNet().to(self.device)
        self.model.apply(weight_init)
        self.loss = self.select_loss_function()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        if os.path.exists(PROCESSED_PATH):
            self.train_dataloader = load(
                filename=os.path.join(PROCESSED_PATH, "train_dataloader.pkl")
            )
            self.test_dataloader = load(
                filename=os.path.join(PROCESSED_PATH, "test_dataloader.pkl")
            )
        else:
            raise ValueError(
                "Processed data not found. Please run the preprocessing notebook.".capitalize()
            )

    def select_loss_function(self):
        if self.loss == "dice":
            return DiceLoss(smooth=0.01)

    def l1_loss(self, model, lambda_value=0.01):
        return lambda_value * sum(
            (torch.norm(input=params, p=1) for params in model.parameters())
        )

    def l2_loss(self, model, lambda_value=0.01):
        return lambda_value * sum(
            (torch.norm(input=params, p=2) for params in model.parameters())
        )

    def update_training_loss(self, **kwargs):
        self.optimizer.zero_grad()

        train_predicted_masks = self.model(kwargs["images"])
        train_predicted_loss = self.loss(train_predicted_masks, kwargs["masks"])

        train_predicted_loss.backward(retain_graph=True)
        self.optimizer.step()

        return train_predicted_loss.item()

    def update_testing_loss(self, **kwargs):
        test_predicted_masks = self.model(kwargs["images"])
        test_predicted_loss = self.loss(test_predicted_masks, kwargs["masks"])

        return test_predicted_loss.item()

    def saved_checkpoints(self, **kwargs):
        if kwargs["epoch"] != self.epochs:
            if os.path.exists(TRAIN_CHECKPOINT_PATH):
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        TRAIN_CHECKPOINT_PATH,
                        "model_{}.pth".format(kwargs["epoch"] + 1),
                    ),
                )
        else:
            if os.path.exists(TEST_CHECKPOINT_PATH):
                torch.save(
                    self.model.state_dict(),
                    os.path.join(TEST_CHECKPOINT_PATH, "best_model.pth"),
                )

    def show_progress(self, **kwargs):
        if self.is_display == True:
            print(
                "Epochs: [{}/{}] - train_loss: [{:.5f}] - test_loss: [{:.5f}]".format(
                    kwargs["epoch"],
                    kwargs["epochs"],
                    kwargs["train_loss"],
                    kwargs["test_loss"],
                )
            )
        elif self.is_display == False:
            logging.info(
                "Epochs: [{}/{}] - train_loss: [{:.5f}] - test_loss: [{:.5f}]".format(
                    kwargs["epoch"],
                    kwargs["epochs"],
                    kwargs["train_loss"],
                    kwargs["test_loss"],
                )
            )

    def train(self):
        try:
            self.__setup__()
        except Exception as e:
            print("The exception in the section # {}".format(e).capitalize())
        else:
            self.model.train()
            for epoch in tqdm(range(self.epochs)):
                train_loss = list()
                test_loss = list()

                for _, (images, masks) in enumerate(self.train_dataloader):
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    train_loss.append(
                        self.update_training_loss(images=images, masks=masks)
                    )

                self.model.eval()

                for _, (images, masks) in enumerate(self.test_dataloader):
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    test_loss.append(
                        self.update_testing_loss(images=images, masks=masks)
                    )

                try:
                    self.saved_checkpoints(epoch=epoch + 1)

                    self.history["train_loss"].append(np.mean(train_loss))
                    self.history["test_loss"].append(np.mean(test_loss))

                except Exception as e:
                    print(e)
                else:
                    images, _ = next(iter(self.test_dataloader))
                    predicted_masks = self.model(images.to(self.device))
                    if os.path.exists(TRAIN_IMAGES_PATH):
                        save_image(
                            predicted_masks,
                            os.path.join(
                                TRAIN_IMAGES_PATH,
                                "train_masks_{}.png".format(epoch + 1),
                            ),
                            nrow=6,
                            normalize=True,
                        )
                    else:
                        raise Exception("Train images path not found.".capitalize())
                finally:
                    self.show_progress(
                        epoch=epoch + 1,
                        epochs=self.epochs,
                        train_loss=np.mean(train_loss),
                        test_loss=np.mean(test_loss),
                    )

            if os.path.exists(METRICS_PATH):
                dump(
                    value=self.history,
                    filename=os.path.join(METRICS_PATH, "metrics.pkl"),
                )
            else:
                raise Exception("Metrics path not found.".capitalize())

    @staticmethod
    def plot_loss_curves():
        if os.path.exists(METRICS_PATH):
            history = load(filename=os.path.join(METRICS_PATH, "metrics.pkl"))
            plt.figure(figsize=(15, 10))

            plt.plot(history["train_loss"], label="Train Loss".title())
            plt.plot(history["test_loss"], label="Test Loss".title())

            plt.title("Loss Curves".title())
            plt.xlabel("Epochs".title())
            plt.ylabel("Loss".title())
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            raise Exception("Metrics path not found.".capitalize())
