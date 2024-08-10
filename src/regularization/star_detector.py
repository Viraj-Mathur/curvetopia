import numpy as np

def is_star(XY, num_points=5, tolerance=0.05):
    """
    Determines if a given set of points forms a regular star shape.
    
    Parameters:
        XY (numpy array): Array of points representing a polyline.
        num_points (int): Expected number of points of the star (typically 5 for a regular star).
        tolerance (float): Tolerance for checking the regularity of the star shape.
    
    Returns:
        is_star (bool): True if the points form a regular star shape, False otherwise.
    """
    if len(XY) != num_points * 2:
        return False
    
    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)
    
    # Calculate distances from the center to the points (radii)
    centroid = np.mean(XY, axis=0)
    radii = np.array([distance(point, centroid) for point in XY])
    
    # Check that the star alternates between two radii (inner and outer points)
    if len(set(np.round(radii, decimals=2))) != 2:
        return False
    
    # Calculate the angles between consecutive points
    angles = []
    for i in range(len(XY)):
        v1 = XY[i] - centroid
        v2 = XY[(i + 1) % len(XY)] - centroid
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        angle = np.degrees(angle)
        if angle < 0:
            angle += 360
        angles.append(angle)
    
    # Check if angles are approximately the same within tolerance
    expected_angle = 360 / (num_points * 2)
    if not np.allclose(angles, expected_angle, atol=tolerance):
        return False
    
    return True

def detect_stars(paths_XYs, num_points=5, tolerance=0.05):
    """
    Detects regular star shapes from given paths (polylines).
    
    Parameters:
        paths_XYs (list): List of paths, where each path is a numpy array of points.
        num_points (int): Expected number of points of the star.
        tolerance (float): Tolerance for detecting regular star shapes.
    
    Returns:
        stars (list): List of detected stars. Each star is represented by its vertices.
    """
    stars = []

    for path in paths_XYs:
        for XY in path:
            if is_star(XY, num_points, tolerance):
                stars.append(XY)

    return stars

if __name__ == "__main__":
    # Example usage
    from src.data_loader import read_csv
    csv_path = "./src/problems"  # Update this path as needed
    curves = read_csv(csv_path)
    
    detected_stars = detect_stars(curves)
    
    # Print detected stars
    for star in detected_stars:
        print("Detected star with vertices:")
        print(star)
