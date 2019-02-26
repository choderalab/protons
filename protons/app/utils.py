import os


def get_datadir():
    """Returns the data directory of this package"""
    return os.path.join(os.path.dirname(__file__), "data")


class OutdatedFileError(Exception):
    """Exception indicating that the file the user supplied is no longer supported."""

    pass
