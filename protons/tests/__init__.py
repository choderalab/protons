import os

PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_test_data(filename, folder):
    """
    Function to obtain data that is part of the package, such as the calibration systems

    Parameters
    ----------

    filename : str
        the name of the file/path you want to obtain
    folder : str
        The name of the folder it is contained in
    """
    return os.path.join(PACKAGE_ROOT, folder, filename)
