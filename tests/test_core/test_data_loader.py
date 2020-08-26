"""Test core.data"""

import os
import unittest

from core.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Class to unittest DataLoader"""

    # A more general testing scheme would not require an explicit path
    # including my username; that said, for the purposes of this exercise,
    # it is sufficient
    # Additionally note that a more complete testing scheme would keep
    # test files out of the repository, particularly as the data requirements
    # increased, but would be unclear for this exercise
    base_dirpath = (
        '/Users/jesse/repos/datapipeline/tests/fixtures_for_tests/'
        'test_data_loader'
    )

    def test__verify_and_parse_dirpath_data_assumptions(self):
        """Test assumptions 1-3 from DataLoader"""

        for assumption_number in [1, 2, 3]:
            dirpath = os.path.join(self.base_dirpath,
                                   f'invalid_dir{assumption_number}')
            with pytest.raises(AssertionError):
                data_loader = DataLoader(dirpath)

    def test_get_fnames_for_batch(self):
        """Test get_fnames_for_batch returns valid batch of fnames

        Note that there are only 10 image/mask pairs in the valid_dir
        """
        dirpath = os.path.join(self.base_dirpath, 'valid_dir')
        data_loader = DataLoader(dirpath_data=dirpath)

        valid_batch_sizes_to_test = [1, 2, 3, 4]
        for batch_size in valid_batch_sizes_to_test:
            fnames_batch = data_loader._get_fnames_for_batch()
