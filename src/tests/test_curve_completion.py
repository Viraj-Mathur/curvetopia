import unittest
import numpy as np
from src.curve_completion.gap_filler import fill_gaps
from src.curve_completion.occlusion_handler import handle_occlusions

class TestCurveCompletion(unittest.TestCase):

    def setUp(self):
        """
        Set up sample data for testing curve completion functions.
        """
        self.curve_with_gaps = np.array([
            [0, 0], [1, 1], [2, 2], 
            [5, 5], [6, 6], [7, 7]  # Gap between [2,2] and [5,5]
        ])
        
        self.curve_with_occlusions = np.array([
            [0, 0], [1, 1], [2, 2],
            [3, 3], [None, None], [None, None],  # Occluded points
            [6, 6], [7, 7], [8, 8]
        ])

    def test_fill_gaps(self):
        """
        Test gap filling functionality.
        """
        filled_curve = fill_gaps(self.curve_with_gaps, tolerance=0.1)

        # Check that gaps are filled
        expected_filled_curve = np.array([
            [0, 0], [1, 1], [2, 2],
            [3, 3], [4, 4], [5, 5],
            [6, 6], [7, 7]
        ])

        np.testing.assert_array_almost_equal(filled_curve, expected_filled_curve, decimal=2,
                                             err_msg="Gap filling did not produce expected results.")

    def test_handle_occlusions(self):
        """
        Test occlusion handling functionality.
        """
        handled_curve = handle_occlusions(self.curve_with_occlusions)

        # Check that occlusions are handled (filled or interpolated)
        expected_handled_curve = np.array([
            [0, 0], [1, 1], [2, 2],
            [3, 3], [4, 4], [5, 5],  # Interpolated points
            [6, 6], [7, 7], [8, 8]
        ])

        np.testing.assert_array_almost_equal(handled_curve, expected_handled_curve, decimal=2,
                                             err_msg="Occlusion handling did not produce expected results.")

if __name__ == '__main__':
    unittest.main()
