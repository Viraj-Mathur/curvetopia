import unittest
import numpy as np
from src.regularization.line_detector import detect_lines
from src.regularization.circle_detector import detect_circles
from src.regularization.ellipse_detector import detect_ellipses
from src.regularization.rectangle_detector import detect_rectangles
from src.regularization.polygon_detector import detect_polygons
from src.regularization.star_detector import detect_stars

class TestRegularization(unittest.TestCase):

    def setUp(self):
        """
        Set up sample data for testing regularization functions.
        """
        # Example data for each shape type
        self.line_data = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        self.circle_data = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        self.ellipse_data = np.array([[2, 0], [0, 1], [-2, 0], [0, -1]])
        self.rectangle_data = np.array([[0, 0], [0, 2], [3, 2], [3, 0]])
        self.polygon_data = np.array([[0, 0], [1, 2], [2, 0]])
        self.star_data = np.array([[0, 1], [0.5, 0.5], [1, 1], [0.75, 0.25], [1, 0]])

    def test_detect_lines(self):
        """
        Test line detection functionality.
        """
        detected_lines = detect_lines(self.line_data)
        self.assertEqual(len(detected_lines), 1, "Line detection failed.")
        self.assertTrue(np.allclose(detected_lines[0], self.line_data), "Detected line does not match the expected line.")

    def test_detect_circles(self):
        """
        Test circle detection functionality.
        """
        detected_circles = detect_circles(self.circle_data)
        self.assertEqual(len(detected_circles), 1, "Circle detection failed.")
        self.assertTrue(np.allclose(detected_circles[0], self.circle_data, atol=0.1), "Detected circle does not match the expected circle.")

    def test_detect_ellipses(self):
        """
        Test ellipse detection functionality.
        """
        detected_ellipses = detect_ellipses(self.ellipse_data)
        self.assertEqual(len(detected_ellipses), 1, "Ellipse detection failed.")
        self.assertTrue(np.allclose(detected_ellipses[0], self.ellipse_data, atol=0.1), "Detected ellipse does not match the expected ellipse.")

    def test_detect_rectangles(self):
        """
        Test rectangle detection functionality.
        """
        detected_rectangles = detect_rectangles(self.rectangle_data)
        self.assertEqual(len(detected_rectangles), 1, "Rectangle detection failed.")
        self.assertTrue(np.allclose(detected_rectangles[0], self.rectangle_data, atol=0.1), "Detected rectangle does not match the expected rectangle.")

    def test_detect_polygons(self):
        """
        Test polygon detection functionality.
        """
        detected_polygons = detect_polygons(self.polygon_data)
        self.assertEqual(len(detected_polygons), 1, "Polygon detection failed.")
        self.assertTrue(np.allclose(detected_polygons[0], self.polygon_data, atol=0.1), "Detected polygon does not match the expected polygon.")

    def test_detect_stars(self):
        """
        Test star detection functionality.
        """
        detected_stars = detect_stars(self.star_data)
        self.assertEqual(len(detected_stars), 1, "Star detection failed.")
        self.assertTrue(np.allclose(detected_stars[0], self.star_data, atol=0.1), "Detected star does not match the expected star.")

if __name__ == '__main__':
    unittest.main()
