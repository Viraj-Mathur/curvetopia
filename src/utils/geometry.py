import numpy as np

def distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    
    Parameters:
        point1 (tuple or np.array): The first point (x1, y1).
        point2 (tuple or np.array): The second point (x2, y2).
    
    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def is_line(points, tolerance=1e-6):
    """
    Check if a sequence of points forms a straight line.
    
    Parameters:
        points (np.array): Array of points [(x1, y1), (x2, y2), ...].
        tolerance (float): The maximum allowed deviation from the straight line.
    
    Returns:
        bool: True if the points form a line, False otherwise.
    """
    if len(points) < 2:
        return False
    
    # Vector from the first to the last point
    vector = points[-1] - points[0]
    norm_vector = np.linalg.norm(vector)
    
    if norm_vector < tolerance:
        return False  # Degenerate case: all points are the same
    
    # Normalize the vector
    unit_vector = vector / norm_vector
    
    # Check the orthogonal distance of each intermediate point from the line
    for point in points[1:-1]:
        distance_to_line = np.linalg.norm(np.cross(unit_vector, point - points[0]))
        if distance_to_line > tolerance:
            return False
    
    return True

def fit_circle(points):
    """
    Fit a circle to a given set of points using the least squares method.
    
    Parameters:
        points (np.array): Array of points [(x1, y1), (x2, y2), ...].
    
    Returns:
        tuple: Center (x, y) and radius of the fitted circle.
    """
    # Formulate the system of equations to solve for center and radius
    A = np.array([[2*point[0], 2*point[1], 1] for point in points])
    b = np.array([point[0]**2 + point[1]**2 for point in points])
    
    # Solve the least squares problem
    c, d, e = np.linalg.lstsq(A, b, rcond=None)[0]
    
    center = np.array([c, d])
    radius = np.sqrt(e + center.dot(center))
    
    return center, radius

def fit_ellipse(points):
    """
    Fit an ellipse to a given set of points.
    
    Parameters:
        points (np.array): Array of points [(x1, y1), (x2, y2), ...].
    
    Returns:
        tuple: Parameters of the fitted ellipse (center_x, center_y, axis1, axis2, angle).
    """
    # Placeholder implementation, replace with an actual fitting algorithm.
    raise NotImplementedError("Ellipse fitting is not yet implemented.")

def is_regular_polygon(points, sides, tolerance=1e-6):
    """
    Check if the given points form a regular polygon with a specified number of sides.
    
    Parameters:
        points (np.array): Array of points [(x1, y1), (x2, y2), ...].
        sides (int): Number of sides for the polygon.
        tolerance (float): Maximum allowed deviation from regularity.
    
    Returns:
        bool: True if the points form a regular polygon, False otherwise.
    """
    if len(points) != sides:
        return False
    
    # Calculate all side lengths
    side_lengths = [distance(points[i], points[(i+1) % sides]) for i in range(sides)]
    
    # Check if all sides are of equal length
    if np.std(side_lengths) > tolerance:
        return False
    
    # Calculate all angles between adjacent sides
    angles = []
    for i in range(sides):
        v1 = points[i] - points[(i-1) % sides]
        v2 = points[(i+1) % sides] - points[i]
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(angle)
    
    # Check if all angles are equal
    if np.std(angles) > tolerance:
        return False
    
    return True

def polygon_area(points):
    """
    Calculate the area of a polygon given its vertices using the shoelace formula.
    
    Parameters:
        points (np.array): Array of points [(x1, y1), (x2, y2), ...].
    
    Returns:
        float: The area of the polygon.
    """
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
