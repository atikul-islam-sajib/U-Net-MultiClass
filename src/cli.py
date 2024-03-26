import argparse
import sys

sys.path.append("src/")

from dataloader import Loader
from UNet import UNet
from trainer import Trainer
from test import Charts


def cli():
    parser = argparse.ArgumentParser(
        description="A CLI tool to train and testing the model - UNet".title()
    )
    parser.add_argument(
        "--image_path", type=str, help="Path to the zip file".capitalize()
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for the dataloader".capitalize(),
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.25,
        help="Split the dataset into train and test sets".capitalize(),
    )
    parser.add_argument(
        "--image_size", type=int, default=128, help="Image size".capitalize()
    )
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
    parser.add_argument("--test", action="store_true", help="Train model".capitalize())

    args = parser.parse_args()

    if args.train:
        if (
            args.image_path
            and args.batch_size
            and args.split_ratio
            and args.image_size
            and args.epochs
            and args.lr
            and args.loss
            and args.display
            and args.device
            and args.smooth_value
            and args.alpha
            and args.gamma
        ):
            loader = Loader(
                image_path=args.image_path,
                batch_size=args.batch_size,
                split_ratio=args.split_ratio,
                image_size=args.image_size,
            )
            loader.unzip_folder()
            loader.create_dataloader()

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

    elif args.test:
        if args.device:
            charts = Charts(device="mps")
            charts.test()


if __name__ == "__main__":
    cli()
