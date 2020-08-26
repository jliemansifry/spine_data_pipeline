"""Test SpineDataLoader"""

import os
import unittest

from spine.data_loader import SpineDataLoader


class TestSpineDataLoader(unittest.TestCase):
    """Class to test SpineDataLoader"""

    dirpath = (
        '/Users/jesse/repos/datapipeline/tests/fixtures_for_tests/'
        'test_data_loader/valid_dir'
    )

    def test_get_image(self):
        """Test get_image returns raw image with expected properties"""

        spine_data_loader = SpineDataLoader(dirpath_data=self.dirpath,
                                            batch_size=4)

        for idx in range(4):
            image = spine_data_loader.get_image(str(idx))
            assert image.shape == (256, 256, 1)
            assert image.min() == 0.0
            assert image.max() == 1.0
            assert image.dtype == 'float64'

    def test_get_mask(self):
        """Test get_mask returns a mask with expected properties"""

        spine_data_loader = SpineDataLoader(dirpath_data=self.dirpath,
                                            batch_size=4)

        for idx in range(4):
            mask = spine_data_loader.get_mask(str(idx))
            assert mask.shape == (256, 256, 1)
            assert mask.dtype == 'int64'
