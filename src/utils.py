# Utils file
import joblib


def dump(value=None, filename=None):
    if value is not None and filename is not None:
        joblib.dump(value=value, filename=filename)

    else:
        raise ValueError("value and filename cannot be None".capitalize())


def load(filename):
    if os.path.exists(filename):
        return joblib.load(filename=filename)
    else:
        raise Exception("File not found".capitalize())
