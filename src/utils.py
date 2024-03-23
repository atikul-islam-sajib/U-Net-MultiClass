# Utils file
import os
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
