import numpy as np

def rotate_point(point, angle, center):
    """
    Rotates a point around a center by a given angle.
    
    Parameters:
        point (numpy array): The point to rotate.
        angle (float): The angle of rotation in radians.
        center (numpy array): The center of rotation.
    
    Returns:
        rotated_point (numpy array): The rotated point.
    """
    # Vectorize point and center
    p = np.array(point)
    c = np.array(center)
    
    # Translate point back to origin
    translated_point = p - c
    
    # Rotation matrix
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    # Apply rotation
    rotated_point = np.dot(rotation_matrix, translated_point) + c
    return rotated_point

def check_rotational_symmetry(XY, tolerance=0.01):
    """
    Checks if a given set of points exhibits rotational symmetry.
    
    Parameters:
        XY (numpy array): Array of points representing a polyline.
        tolerance (float): Tolerance for symmetry detection.
    
    Returns:
        bool: True if the points exhibit rotational symmetry, False otherwise.
    """
    n = len(XY)
    if n < 3:
        return False
    
    centroid = np.mean(XY, axis=0)
    
    # Try different rotation angles that could indicate symmetry
    for angle in np.linspace(0, 2 * np.pi, n, endpoint=False):
        rotated_points = []
        for point in XY:
            rotated_point = rotate_point(point, angle, centroid)
            rotated_points.append(rotated_point)
        
        # Vectorized distance calculation between original and rotated points
        distances = np.linalg.norm(XY - np.array(rotated_points), axis=1)
        
        # Check if all points are within the given tolerance
        if np.allclose(distances, 0, atol=tolerance):
            return True
    
    return False

def detect_rotational_symmetries(paths_XYs, tolerance=0.01):
    """
    Detects rotational symmetry in given paths (polylines).
    
    Parameters:
        paths_XYs (list): List of paths, where each path is a numpy array of points.
        tolerance (float): Tolerance for detecting rotational symmetry.
    
    Returns:
        symmetric_paths (list): List of paths that exhibit rotational symmetry.
    """
    symmetric_paths = []

    for path in paths_XYs:
        for XY in path:
            if check_rotational_symmetry(XY, tolerance):
                symmetric_paths.append(XY)

    return symmetric_paths

if __name__ == "__main__":
    # Example usage
    from src.data_loader import read_csv
    csv_path = "./src/problems/frag0.csv"  # Update this path as needed
    curves = read_csv(csv_path)
    
    symmetries = detect_rotational_symmetries(curves)
    
    # Print detected rotational symmetries
    for symmetry in symmetries:
        print("Detected rotational symmetry in the path:")
        print(symmetry)
