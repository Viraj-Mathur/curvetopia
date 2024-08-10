import numpy as np
from scipy.spatial import ConvexHull

def is_polygon(XY, tolerance=0.01):
    """
    Determines if a given set of points forms a regular polygon.
    
    Parameters:
        XY (numpy array): Array of points representing a polyline.
        tolerance (float): Tolerance for checking regularity of sides and angles.
    
    Returns:
        is_polygon (bool): True if the points form a regular polygon, False otherwise.
    """
    if len(XY) < 3:  # A polygon must have at least 3 sides
        return False
    
    # Compute the convex hull of the points
    hull = ConvexHull(XY)
    hull_points = XY[hull.vertices]

    # Check if the number of hull points matches the number of points in the array (simple polygon)
    if len(hull_points) != len(XY):
        return False

    # Calculate distances between consecutive points
    side_lengths = np.sqrt(np.sum(np.diff(hull_points, axis=0)**2, axis=1))
    side_lengths = np.append(side_lengths, np.linalg.norm(hull_points[0] - hull_points[-1]))

    # Calculate angles between consecutive sides using the dot product
    vectors = np.diff(hull_points, axis=0)
    vectors = np.vstack((vectors, hull_points[0] - hull_points[-1]))

    angles = []
    for i in range(len(vectors)):
        vec1 = vectors[i]
        vec2 = vectors[(i+1) % len(vectors)]
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angles.append(np.degrees(angle))
    
    # Check for equal side lengths and angles within the given tolerance
    if np.allclose(side_lengths, side_lengths[0], atol=tolerance) and \
       np.allclose(angles, angles[0], atol=tolerance):
        return True
    
    return False

def detect_polygons(paths_XYs, tolerance=0.01):
    """
    Detects regular polygons from given paths (polylines).
    
    Parameters:
        paths_XYs (list): List of paths, where each path is a numpy array of points.
        tolerance (float): Tolerance for detecting regular polygons.
    
    Returns:
        polygons (list): List of detected polygons. Each polygon is represented by its vertices.
    """
    polygons = []

    for path in paths_XYs:
        for XY in path:
            if is_polygon(XY, tolerance):
                polygons.append(XY)

    return polygons

if __name__ == "__main__":
    # Example usage
    from src.data_loader import read_csv
    csv_path = "./src/problems"  # Update this path as needed
    curves = read_csv(csv_path)
    
    detected_polygons = detect_polygons(curves)
    
    # Print detected polygons
    for polygon in detected_polygons:
        print("Detected polygon with vertices:")
        print(polygon)
