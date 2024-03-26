# Utils file
import os
import warnings
import torch
import torch.nn as nn
import joblib


def dump(value=None, filename=None):
    """
    Serializes and saves a Python object to a file using Joblib.

    Parameters:
    - value: The Python object to serialize and save. Cannot be None.
    - filename: The path to the file where the object should be saved. Cannot be None.

    Raises:
    - ValueError: If either 'value' or 'filename' is None.
    """
    if value is not None and filename is not None:
        joblib.dump(value=value, filename=filename)

    else:
        raise ValueError("value and filename cannot be None".capitalize())


def load(filename):
    """
    Loads and deserializes a Python object from a file using Joblib.

    Parameters:
    - filename: The path to the file from which the object should be loaded.

    Returns:
    - The deserialized Python object.

    Raises:
    - Exception: If the specified file does not exist.
    """
    if os.path.exists(filename):
        return joblib.load(filename=filename)
    else:
        raise Exception("File not found".capitalize())


def define_device(device="mps"):
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def weight_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def ignore_warnings():
    warnings.filterwarnings("ignore")
