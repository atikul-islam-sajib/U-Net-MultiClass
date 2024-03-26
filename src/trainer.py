import sys
import os
import logging
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

sys.path.append("src/")

from config import (
    PROCESSED_PATH,
    TRAIN_CHECKPOINT_PATH,
    TEST_CHECKPOINT_PATH,
    TRAIN_IMAGES_PATH,
    METRICS_PATH,
)
from utils import load, dump, weight_init, define_device
from UNet import UNet
from dice_loss import DiceLoss
from tversky_loss import IoU
from focal_loss import FocalLoss


class Trainer:
    """
    A class responsible for training a U-Net model for image segmentation tasks.

    Attributes:
        epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        loss (str): Name of the loss function to be used. Currently supports 'dice' loss.
        smooth_value (float): Smoothing value for the Dice loss function to avoid division by zero.
        beta1 (float): The exponential decay rate for the first moment estimates in optimizer.
        device (str): Device on which to train the model (e.g., 'cpu', 'cuda', 'mps').
        is_display (bool): Flag to display progress during training.
        history (dict): A dictionary to record training and testing loss over epochs.

    Methods:
        __setup__(): Sets up the model, loss function, optimizer, and data loaders.
        select_loss_function(): Selects the loss function based on the 'loss' attribute.
        l1_loss(model, lambda_value=0.01): Calculates the L1 loss for regularization.
        l2_loss(model, lambda_value=0.01): Calculates the L2 loss for regularization.
        update_training_loss(**kwargs): Performs a single training step.
        update_testing_loss(**kwargs): Evaluates the model on the test set.
        saved_checkpoints(**kwargs): Saves model checkpoints during training.
        show_progress(**kwargs): Prints or logs the training progress.
        train(): Main method to run the training loop.
        plot_loss_curves(): Static method to plot the training and testing loss curves.

    Examples:
        >>> trainer = Trainer(epochs=10, lr=0.001, loss='dice', device='cuda')
        >>> trainer.train()
    """

    def __init__(
        self,
        epochs=10,
        lr=0.0002,
        loss="dice",
        alpha=0.25,
        gamma=2,
        smooth_value=0.01,
        beta1=0.5,
        device="mps",
        display=True,
    ):
        self.epochs = epochs
        self.lr = lr
        self.loss = loss
        self.alpha = alpha
        self.gamma = gamma
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
        """
        Selects and returns the loss function based on the loss type specified during initialization.

        Returns:
            nn.Module: The loss function as a PyTorch module.
        """
        if self.loss == "dice":
            return DiceLoss(smooth=0.01)
        elif self.loss == "IoU":
            return IoU(smooth=self.smooth_value)
        elif self.loss == "focal":
            return FocalLoss(alpha=self.alpha, gamma=self.gamma)
        else:
            raise ValueError(
                "Loss function not supported. Please choose from 'dice' or 'IoU'.".capitalize()
            )

    def l1_loss(self, model, lambda_value=0.01):
        """
        Calculates the L1 regularization loss.

        Parameters:
            model (torch.nn.Module): The model whose weights are regularized.
            lambda_value (float): Regularization coefficient.

        Returns:
            torch.Tensor: The L1 regularization loss.
        """
        return lambda_value * sum(
            (torch.norm(input=params, p=1) for params in model.parameters())
        )

    def l2_loss(self, model, lambda_value=0.01):
        """
        Calculates the L2 regularization loss.

        Parameters:
            model (torch.nn.Module): The model whose weights are regularized.
            lambda_value (float): Regularization coefficient.

        Returns:
            torch.Tensor: The L2 regularization loss.
        """
        return lambda_value * sum(
            (torch.norm(input=params, p=2) for params in model.parameters())
        )

    def update_training_loss(self, **kwargs):
        """
        Updates the model's weights by performing a single step of training.

        Parameters:
            kwargs (dict): Contains 'images' and 'masks', both of type torch.Tensor, representing
                           the input images and their corresponding ground truth masks.

        Returns:
            float: The training loss for the current step.
        """
        self.optimizer.zero_grad()

        train_predicted_masks = self.model(kwargs["images"])
        train_predicted_loss = self.loss(train_predicted_masks, kwargs["masks"])

        train_predicted_loss.backward(retain_graph=True)
        self.optimizer.step()

        return train_predicted_loss.item()

    def update_testing_loss(self, **kwargs):
        """
        Computes the loss on the test dataset without updating the model's weights.

        Parameters:
            kwargs (dict): Contains 'images' and 'masks', both of type torch.Tensor, representing
                           the input images and their corresponding ground truth masks.

        Returns:
            float: The testing loss for the current step.
        """
        test_predicted_masks = self.model(kwargs["images"])
        test_predicted_loss = self.loss(test_predicted_masks, kwargs["masks"])

        return test_predicted_loss.item()

    def saved_checkpoints(self, **kwargs):
        """
        Saves the model's checkpoints at specified intervals.

        Parameters:
            kwargs (dict): Contains 'epoch', the current epoch during training.
        """
        if kwargs["epoch"] != self.epochs:
            if os.path.exists(TRAIN_CHECKPOINT_PATH):
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        TRAIN_CHECKPOINT_PATH,
                        "model_{}.pth".format(kwargs["epoch"]),
                    ),
                )
        else:
            if os.path.exists(TEST_CHECKPOINT_PATH):
                torch.save(
                    self.model.state_dict(),
                    os.path.join(TEST_CHECKPOINT_PATH, "best_model.pth"),
                )

    def show_progress(self, **kwargs):
        """
        Displays or logs the training progress, including the current epoch and loss.

        Parameters:
            kwargs (dict): Contains 'epoch', 'epochs', 'train_loss', and 'test_loss', detailing
                           the current epoch, total epochs, training loss, and testing loss, respectively.
        """
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
        """
        The main method for training the model. It sets up the model, iterates over the dataset
        for a given number of epochs, updates the model weights, and saves the progress.
        """
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
        """
        Plots the training and testing loss curves. Requires the metrics to be saved in a specified path.

        Raises:
            Exception: If the metrics path is not found.
        """
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer".title())
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs".capitalize()
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="Learning rate".capitalize()
    )
    parser.add_argument(
        "--loss", type=str, default="dice", help="Loss function".capitalize()
    )
    parser.add_argument(
        "--display", type=bool, default=True, help="Display progress".capitalize()
    )
    parser.add_argument("--device", type=str, default="mps", help="Device".capitalize())
    parser.add_argument(
        "--smooth_value", type=float, default=0.01, help="Smooth value".capitalize()
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha value".capitalize()
    )
    parser.add_argument(
        "--gamma", type=float, default=2, help="Gamma value".capitalize()
    )
    parser.add_argument("--train", action="store_true", help="Train model".capitalize())

    args = parser.parse_args()

    if args.train:
        if (
            args.epochs
            and args.lr
            and args.loss
            and args.display
            and args.device
            and args.smooth_value
            and args.alpha
            and args.gamma
        ):
            trainer = Trainer(
                epochs=args.epochs,
                lr=args.lr,
                loss=args.loss,
                alpha=args.alpha,
                gamma=args.gamma,
                display=args.display,
                device=args.device,
                smooth_value=args.smooth_value,
            )
            trainer.train()
    else:
        raise Exception("Train flag is not set.".capitalize())
