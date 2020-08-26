"""Data Loader for spine images and masks"""

import imageio
import os
import numpy as np

from core.data_loader import DataLoader


class SpineDataLoader(DataLoader):
    """Class for loading spine images and corresponding masks"""

    def __init__(self, dirpath_data, batch_size):
        """Init

        :param str dirpath_data: the directory path to the spine
        images and masks
        :param int batch_size: the size of the batch
        """

        super().__init__(dirpath_data, batch_size)

    def get_image(self, fname):
        """Get spine image from corresponding image filename index

        :param str fname: the filename, i.e. X in X.png or X.dcm
        :return: image, of shape (256, 256, 1), normalized to 0-1
        """

        fname = os.path.join(self.dirpath_data, 'images', fname)
        image = imageio.imread(f'{fname}.png') / 255
        image = image[..., np.newaxis]

        return image

    def get_mask(self, fname):
        """Get spine image from corresponding mask filename index

        :param str fname: the filename, i.e. Y in Y.npy
        :return: mask, of shape (256, 256, 1), with values of 0 corresponding
        to background, 1 corresponding to disc, and 2 corresponding
        to vertebrae
        """

        fname = os.path.join(self.dirpath_data, 'masks', fname)
        mask = np.load(f'{fname}.npy')
        mask = mask[..., np.newaxis]

        return mask
