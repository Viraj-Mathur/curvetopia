import unittest
import numpy as np
from src.data_loader import read_csv

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        """
        Set up paths and expected data for testing data loading functions.
        """
        self.csv_path = "./src/problems/frag0.csv"  # Update this path as needed
        
        # Expected data format after loading
        self.expected_data = np.array([
            [0, 0], [1, 1], [2, 2],
            [3, 3], [4, 4], [5, 5]
        ])

    def test_read_csv(self):
        """
        Test CSV data loading functionality.
        """
        loaded_data = read_csv(self.csv_path)

        # Ensure data types are correct
        self.assertIsInstance(loaded_data, np.ndarray, "Loaded data is not a NumPy array.")
        self.assertEqual(loaded_data.shape[1], 2, "Loaded data does not have two columns (x, y).")
        
        # Check if loaded data matches expected data
        np.testing.assert_array_almost_equal(loaded_data, self.expected_data, decimal=2,
                                             err_msg="Loaded data does not match expected results.")

    def test_data_integrity(self):
        """
        Test integrity of loaded data (no NaNs or infinite values).
        """
        loaded_data = read_csv(self.csv_path)
        
        # Check for NaNs
        self.assertFalse(np.isnan(loaded_data).any(), "Loaded data contains NaN values.")
        
        # Check for infinite values
        self.assertFalse(np.isinf(loaded_data).any(), "Loaded data contains infinite values.")

if __name__ == '__main__':
    unittest.main()
