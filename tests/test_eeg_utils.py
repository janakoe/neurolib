import unittest
from neurolib.models.eeg.eeg_utils import access_atlases
from neurolib.models.eeg.eeg_utils import downsample_leadfield_matrix
import numpy as np


class TestAccessAtlases(unittest.TestCase):

    def test_get_labels_of_points_aal2(self):

        # An arbitrary testing point is the MNI-coordinate (-6, 14, 0). E.g. mentioned in doi:10.1093/brain/awt329
        # that the left caudate nucleus is located there.
        test_point = np.array([[-6, 14, 0]])    # One sample point with known label.
        result = access_atlases.get_labels_of_points(test_point, atlas="aal2")
        expected_result = ([True], np.array([7001, ]), ["Caudate_L"])
        self.assertTupleEqual(expected_result, result)

        # A point that should be way out of the space covered by any brain-atlas.
        expected_result = ([False], np.array([np.NAN, ]), ["invalid"])
        test_point = np.array([[1000, 0, 0]])
        points_valid, codes, acronyms = access_atlases.get_labels_of_points(test_point, atlas="aal2")
        self.assertEqual(expected_result[0], points_valid)
        assert np.isnan(codes[0])
        self.assertEqual(expected_result[2], acronyms)

    def test_filter_for_regions(self):
        regions = ["abc", "def"]
        labels = ["0", 0, np.NAN, "abc", "abcdef", "ABC"]
        self.assertListEqual(access_atlases.filter_for_regions(labels, regions),
                             [False, False, False, True, False, False])


class TestDownsampleLeadfield(unittest.TestCase):

    def test_downsample_leadfield_matrix(self):
        test_matrix = np.repeat(np.arange(0, 10, 1), 6).reshape((-1, 6))  # ten channels, six source locations
        test_matrix[:, 1] = -test_matrix[:, 1]      # make dipoles of first region cancel each other out
        test_matrix[:, 2] = test_matrix[:, 2] + 2   #

        label_codes = np.array((0, 1, 2, 3, 1, 2))  # First source is in not-of-interest area, the other five sources
                                                    # fall into three different regions.

        unique_labels, downsampled_leadfield = downsample_leadfield_matrix(test_matrix, label_codes)

        self.assertTrue(np.all(downsampled_leadfield.shape == (10, 3)))

        expected_results = {1: np.zeros(10), 2: np.arange(0, 10, 1)+1, 3: np.arange(0, 10, 1)}

        for idx_label, label in enumerate(unique_labels):
            self.assertTrue(np.all(expected_results[label] == downsampled_leadfield[:, idx_label]))
