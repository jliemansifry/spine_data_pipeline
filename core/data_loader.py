"""Data Loader and Pipeline Handler"""

import os
import random
import numpy as np

from core.utilities import get_filenames_and_exts


class DataLoader():
    """General class to load images and corresponding masks

    Assumptions for simplicity that could be generalized further if desired:
    (1) dirpath_data contains only folders 'images' and 'masks'
    (2) the filenames of the files in 'images' and 'masks' are identical,
      i.e. there is a 1:1 correspondence between image data and mask data
      as identified by the filename
      (While the code could prune or warm about mismatching images/masks;
       for the purposes of this exercise, any mismatch will
       result in a failed assert)
    (3) all files in a given directory are of the same file type
    (4) if the number of examples is not divisible by the batch_size,
      the necessary number of surplus examples will be randomly selected from
      the rest of the available examples to fill out the batch;
      as the dataset will be shuffled each epoch, this will not result in
      a systematic oversampling of any specific examples
    (5) the batch size must be smaller than the number of training examples

    Note: for simplicity, this class assumes images and masks
    are stored in the filesystem on disk (rather than in a SQL
    database or some other type of database).
    """

    def __init__(self, dirpath_data, batch_size=4):
        """Init

        :param str dirpath_data: the directory path to the
        folder holding the images and masks
        :param int batch_size: the size of the batch
        """

        self.valid_fnames = self._verify_and_parse_dirpath_data(dirpath_data)
        self.num_valid_fnames = len(self.valid_fnames)
        self.train_fnames, self.val_fnames, self.test_fnames = (
            self._split_to_train_val_test()
        )
        self.batch_size = batch_size
        self.num_train_steps = int(len(self.train_fnames) / self.batch_size)
        self.num_val_steps = int(len(self.val_fnames) / self.batch_size)
        self.fnames_counter = 0
        self.dirpath_data = dirpath_data

    def _split_to_train_val_test(self):
        """Assign valid_fnames into train, val and test sets

        Note: this framework was sufficient for this exercise, but modular
        dataset-specifying code (e.g. specifying train/val/
        test split fractions)
        as well as data augmentations, etc., and utilizing pandas DataFrame
        objects could be considered for future work.

        :return: tuple of:
        - list[str] train_fnames: the filenames for the training set
        - list[str] val_fnames: the filenames for the val set
        - list[str] test_fnames: the filenames for the test set
        """

        train_split, val_split, test_split = 0.6, 0.2, 0.2
        num_train = self.num_valid_fnames * train_split
        num_val = self.num_valid_fnames * val_split
        num_test = self.num_valid_fnames * test_split

        for num_set in [num_train, num_val, num_test]:
            # assert there was a clean split of data; this could be
            # handled more completely but is sufficient for this exercise
            assert num_set % 1 == 0

        random.shuffle(self.valid_fnames)
        train_fnames, val_fnames, test_fnames = (
            np.split(self.valid_fnames,
                     [int(num_train), int(num_train + num_val)])
        )

        return train_fnames, val_fnames, test_fnames

    def _verify_and_parse_dirpath_data(self, dirpath_data):
        """Verify that the dirpath given has expected properties

        Expected properties are assumptions 1, 2, and 3 in the class docstring

        :param str dirpath_data: the directory path to the 'images' and 'masks'
        :return:
        - list[str] valid_filenames: the filenames that are verified to exist
        in both 'images' and 'masks' directories
        """

        # assumption/check (1)
        expected_directories = ['images', 'masks']
        found_directories = [dir for dir in os.listdir(dirpath_data)
                             if not dir.startswith('.')]
        assert sorted(found_directories) == expected_directories

        # assumption/check (2)
        fnames_images, exts_images = (
            get_filenames_and_exts(os.path.join(dirpath_data, 'images'))
        )
        fnames_masks, exts_masks = (
            get_filenames_and_exts(os.path.join(dirpath_data, 'masks'))
        )
        assert sorted(fnames_images) == sorted(fnames_masks)
        valid_filenames = fnames_images

        # assumption/check (3)
        assert len(np.unique(exts_images)) == 1
        assert len(np.unique(exts_masks)) == 1

        return valid_filenames

    def _get_fnames_for_batch(self, set_name='train'):
        """Get the filenames for a batch

        :param str set_name: the name of the set to get fnames for,
        one of 'train', 'val', 'test'
        :return:
        - list[str] fnames_batch: the list of filenames for the batch
        """

        assert set_name in ['train', 'val', 'test']
        num_set_fnames = len(getattr(self, f'{set_name}_fnames'))

        assert self.batch_size <= num_set_fnames

        idx_start = self.fnames_counter
        idx_end = idx_start + self.batch_size
        self.fnames_counter += self.batch_size

        final_batch = self.fnames_counter >= num_set_fnames
        if final_batch:
            overflow_count = idx_end - num_set_fnames
            fnames_overflow = np.random.choice(
                self.valid_fnames[:idx_start], overflow_count, replace=False
            )
            idx_end = num_set_fnames

        fnames_batch = np.array(self.valid_fnames[idx_start:idx_end])

        if final_batch:
            fnames_batch = np.concatenate([fnames_batch, fnames_overflow])
            self.fnames_counter = 0
            random.shuffle(self.train_fnames)

        return fnames_batch

    def get_image(self, fname):
        """Get image via image filename

        Note that this is not implemented in this general class
        such that derived classes could make accommodations depending
        on if images are .png, DICOM, etc.

        :param str fname: the filename, i.e. X in X.png or X.dcm
        :return: image, with dimensions consistent with choice of network
        """

        raise NotImplementedError

    def get_mask(self, fname_idx):
        """Get image via mask filename

        :param str fname: the filename, i.e. Y in Y.npy
        :return: mask, with dimensions consistent with choice of network
        and activation
        """

        raise NotImplementedError

    def get_batch(self, set_name='train'):
        """Get a batch of images and masks

        :param str set_name: the name of the set to get fnames for,
        one of 'train', 'val', 'test'
        :yield:
        - np.array images, a batch worth of images for the network
        - np.array masks, a batch worth of masks for the network
        """

        while True:
            fnames_batch = self._get_fnames_for_batch(set_name=set_name)
            images = np.array([self.get_image(fname)
                               for fname in fnames_batch])
            masks = np.array([self.get_mask(fname)
                              for fname in fnames_batch])

            yield images, masks
