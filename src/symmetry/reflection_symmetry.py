import numpy as np

def reflect_point(point, line_point1, line_point2):
    """
    Reflects a point across a line defined by two points.
    
    Parameters:
        point (numpy array): The point to reflect.
        line_point1 (numpy array): The first point on the line.
        line_point2 (numpy array): The second point on the line.
    
    Returns:
        reflected_point (numpy array): The reflected point.
    """
    # Vectorize points
    p1 = np.array(line_point1)
    p2 = np.array(line_point2)
    p = np.array(point)
    
    # Vector from line_point1 to line_point2
    d = p2 - p1
    
    # Project point onto the line
    projection = p1 + np.dot(p - p1, d) / np.dot(d, d) * d
    
    # Reflection formula
    reflected_point = 2 * projection - p
    return reflected_point

def check_reflection_symmetry(XY, tolerance=0.01):
    """
    Checks if a given set of points exhibits reflectional symmetry.
    
    Parameters:
        XY (numpy array): Array of points representing a polyline.
        tolerance (float): Tolerance for symmetry detection.
    
    Returns:
        bool: True if the points exhibit reflectional symmetry, False otherwise.
    """
    n = len(XY)
    if n < 2:
        return False
    
    # Try different pairs of points as potential axes of symmetry
    for i in range(n):
        for j in range(i + 1, n):
            reflected_points = []
            for point in XY:
                reflected_point = reflect_point(point, XY[i], XY[j])
                reflected_points.append(reflected_point)
            
            # Vectorized distance calculation between original and reflected points
            distances = np.linalg.norm(XY - np.array(reflected_points), axis=1)
            
            # Check if all points are within the given tolerance
            if np.allclose(distances, 0, atol=tolerance):
                return True
    
    return False

def detect_reflection_symmetries(paths_XYs, tolerance=0.01):
    """
    Detects reflectional symmetry in given paths (polylines).
    
    Parameters:
        paths_XYs (list): List of paths, where each path is a numpy array of points.
        tolerance (float): Tolerance for detecting reflectional symmetry.
    
    Returns:
        symmetric_paths (list): List of paths that exhibit reflectional symmetry.
    """
    symmetric_paths = []

    for path in paths_XYs:
        for XY in path:
            if check_reflection_symmetry(XY, tolerance):
                symmetric_paths.append(XY)

    return symmetric_paths

if __name__ == "__main__":
    # Example usage
    from src.data_loader import read_csv
    csv_path = "./src/problems/frag0.csv"  # Update this path as needed
    curves = read_csv(csv_path)
    
    symmetries = detect_reflection_symmetries(curves)
    
    # Print detected symmetries
    for symmetry in symmetries:
        print("Detected reflectional symmetry in the path:")
        print(symmetry)
