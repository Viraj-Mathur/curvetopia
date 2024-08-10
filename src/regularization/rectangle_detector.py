import numpy as np

def is_rectangle(XY, tolerance=0.01):
    """
    Determines if a given set of points forms a rectangle.
    
    Parameters:
        XY (numpy array): Array of points representing a polyline.
        tolerance (float): Tolerance for checking the right angles and parallel sides.
    
    Returns:
        is_rectangle (bool): True if the points form a rectangle, False otherwise.
    """
    if len(XY) != 4:  # A rectangle must have exactly 4 vertices
        return False

    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)
    
    def is_right_angle(v1, v2):
        dot_product = np.dot(v1, v2)
        return np.isclose(dot_product, 0, atol=tolerance)
    
    # Calculate distances between consecutive points
    dists = [distance(XY[i], XY[(i + 1) % 4]) for i in range(4)]
    
    # Check if opposite sides are equal
    if not np.isclose(dists[0], dists[2], atol=tolerance) or \
       not np.isclose(dists[1], dists[3], atol=tolerance):
        return False
    
    # Calculate vectors for the sides
    vectors = [XY[(i + 1) % 4] - XY[i] for i in range(4)]
    
    # Check if all angles are right angles
    for i in range(4):
        if not is_right_angle(vectors[i], vectors[(i + 1) % 4]):
            return False
    
    return True

def detect_rectangles(paths_XYs, tolerance=0.01):
    """
    Detects rectangles from given paths (polylines).
    
    Parameters:
        paths_XYs (list): List of paths, where each path is a numpy array of points.
        tolerance (float): Tolerance for detecting rectangles.
    
    Returns:
        rectangles (list): List of detected rectangles. Each rectangle is represented by its vertices.
    """
    rectangles = []

    for path in paths_XYs:
        for XY in path:
            if is_rectangle(XY, tolerance):
                rectangles.append(XY)

    return rectangles

if __name__ == "__main__":
    # Example usage
    from src.data_loader import read_csv
    csv_path = "./src/problems"  # Update this path as needed
    curves = read_csv(csv_path)
    
    detected_rectangles = detect_rectangles(curves)
    
    # Print detected rectangles
    for rectangle in detected_rectangles:
        print("Detected rectangle with vertices:")
        print(rectangle)
