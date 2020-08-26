"""Common utilities"""

import os


def get_filenames_and_exts(dirpath):
    """Get the filenames in a directory, without the extensions

    For example, a folder of .npy files called '0.npy', '1.npy', etc.,
    would return ['0', '1', ...]

    :param str dirpath: the directory path to parse for filenames
    :return:
    - list fnames: the filenames in the directory
    """

    fnames = [os.path.splitext(fname)[0] for fname in os.listdir(dirpath)
              if not fname.startswith('.')]
    exts = [os.path.splitext(fname)[1] for fname in os.listdir(dirpath)
            if not fname.startswith('.')]

    return fnames, exts
