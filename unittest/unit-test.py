import sys
import os
import logging
import unittest
import torch

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="unit-test.log",
)

sys.path.append("src/")

from src.utils import load
from src.config import RAW_PATH, PROCESSED_PATH


class UnitTest(unittest.TestCase):
    """
    Defines a suite of unit tests for verifying the integrity and correctness of the
    dataset loading functionality and dataset sizes for training and testing.
    """

    def setUp(self):
        """
        Set up the test environment by loading the train and test dataloaders from
        the PROCESSED_PATH.
        """
        self.train_dataloader = load(
            os.path.join(PROCESSED_PATH, "train_dataloader.pkl")
        )
        self.test_dataloader = load(
            os.path.join(PROCESSED_PATH, "test_data_loader.pkl")
        )

    def test_train_dataset_size(self):
        """
        Test that the dimensions of the data and label tensors in the training dataset
        are as expected.
        """
        data, label = next(iter(self.train_dataloader))
        self.assertEqual(data, torch.Size([4, 3, 256, 256]))
        self.assertEqual(label, torch.Size([4, 1, 256, 256]))

    def test_val_dataset_size(self):
        """
        Test that the dimensions of the data and label tensors in the validation dataset
        are as expected.
        """
        data, label = next(iter(self.test_dataloader))
        self.assertEqual(data, torch.Size([24, 3, 256, 256]))
        self.assertEqual(label, torch.Size([24, 1, 256, 256]))

    def test_quantity_train_dataset(self):
        """
        Test that the total number of samples in the training dataset is as expected.
        """
        self.assertEqual(sum(data.size(0) for data in self.train_dataloader), 3256)

    def test_quantity_val_dataset(self):
        """
        Test that the total number of samples in the validation dataset is as expected.
        """
        self.assertEqual(sum(data.size(0) for data in self.test_dataloader), 3256)
