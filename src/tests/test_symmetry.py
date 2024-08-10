import unittest
import numpy as np
from src.symmetry.reflection_symmetry import detect_reflection_symmetry
from src.symmetry.rotational_symmetry import detect_rotational_symmetry

class TestSymmetry(unittest.TestCase):

    def setUp(self):
        """
        Set up sample data for testing symmetry detection functions.
        """
        # Symmetric along y-axis
        self.reflection_symmetric_data = np.array([
            [-1, 1], [0, 1], [1, 1],
            [-1, 0], [0, 0], [1, 0],
            [-1, -1], [0, -1], [1, -1]
        ])
        
        # 4-fold rotational symmetry (90 degrees)
        self.rotational_symmetric_data = np.array([
            [1, 0], [0, 1], [-1, 0], [0, -1],
            [2, 0], [0, 2], [-2, 0], [0, -2]
        ])

    def test_detect_reflection_symmetry(self):
        """
        Test reflection symmetry detection functionality.
        """
        symmetry_axis, is_symmetric = detect_reflection_symmetry(self.reflection_symmetric_data)
        
        self.assertTrue(is_symmetric, "Reflection symmetry detection failed.")
        self.assertTrue(np.allclose(symmetry_axis, [0, 1], atol=0.1), "Incorrect reflection symmetry axis detected.")

    def test_detect_rotational_symmetry(self):
        """
        Test rotational symmetry detection functionality.
        """
        rotation_center, rotation_order, is_symmetric = detect_rotational_symmetry(self.rotational_symmetric_data)
        
        self.assertTrue(is_symmetric, "Rotational symmetry detection failed.")
        self.assertEqual(rotation_order, 4, "Incorrect rotational symmetry order detected.")
        self.assertTrue(np.allclose(rotation_center, [0, 0], atol=0.1), "Incorrect rotational symmetry center detected.")

if __name__ == '__main__':
    unittest.main()
