import os

PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))

def get_data(path,folder):
    return os.path.join(PACKAGE_ROOT, folder, path)

