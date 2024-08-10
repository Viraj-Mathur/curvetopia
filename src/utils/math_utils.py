import numpy as np

def angle_between_vectors(v1, v2):
    """
    Calculate the angle in radians between two vectors.
    
    Parameters:
        v1 (np.array): The first vector.
        v2 (np.array): The second vector.
    
    Returns:
        float: The angle between the vectors in radians.
    """
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = np.clip(dot_product / norms, -1.0, 1.0)
    return np.arccos(cos_theta)

def rotate_point(point, angle, origin=(0, 0)):
    """
    Rotate a point around a given origin by a specified angle.
    
    Parameters:
        point (tuple): The point (x, y) to rotate.
        angle (float): The angle in radians to rotate the point.
        origin (tuple): The origin (x, y) to rotate around (default is the origin (0, 0)).
    
    Returns:
        tuple: The rotated point (x', y').
    """
    ox, oy = origin
    px, py = point
    
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    
    return qx, qy

def translate_point(point, translation_vector):
    """
    Translate a point by a given vector.
    
    Parameters:
        point (tuple): The point (x, y) to translate.
        translation_vector (tuple): The translation vector (tx, ty).
    
    Returns:
        tuple: The translated point (x', y').
    """
    return point[0] + translation_vector[0], point[1] + translation_vector[1]

def reflect_point(point, axis='x', position=0):
    """
    Reflect a point across a specified axis.

    Parameters:
        point (tuple): The point (x, y) to reflect.
        axis (str): The axis to reflect across ('x' or 'y').
        position (float): The position of the reflection axis (default is 0, i.e., the origin).
    
    Returns:
        tuple: The reflected point (x', y').
    """
    if axis == 'x':
        return point[0], 2 * position - point[1]
    elif axis == 'y':
        return 2 * position - point[0], point[1]
    else:
        raise ValueError("Axis must be 'x' or 'y'.")

def vector_magnitude(vector):
    """
    Calculate the magnitude (length) of a vector.
    
    Parameters:
        vector (np.array): The vector whose magnitude is to be calculated.
    
    Returns:
        float: The magnitude of the vector.
    """
    return np.linalg.norm(vector)

def normalize_vector(vector):
    """
    Normalize a vector to have a magnitude of 1 (unit vector).
    
    Parameters:
        vector (np.array): The vector to normalize.
    
    Returns:
        np.array: The normalized vector.
    """
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector  # Cannot normalize a zero vector
    return vector / magnitude

def vector_projection(v1, v2):
    """
    Project vector v1 onto vector v2.
    
    Parameters:
        v1 (np.array): The vector to be projected.
        v2 (np.array): The vector to project onto.
    
    Returns:
        np.array: The projection of v1 onto v2.
    """
    v2_normalized = normalize_vector(v2)
    return np.dot(v1, v2_normalized) * v2_normalized

def cross_product(v1, v2):
    """
    Compute the cross product of two 2D vectors.
    
    Parameters:
        v1 (np.array): The first vector.
        v2 (np.array): The second vector.
    
    Returns:
        float: The scalar cross product of v1 and v2.
    """
    return v1[0] * v2[1] - v1[1] * v2[0]

def is_point_in_polygon(point, polygon):
    """
    Determine if a point is inside a polygon using the ray-casting algorithm.
    
    Parameters:
        point (tuple): The point (x, y) to check.
        polygon (list of tuples): The vertices of the polygon [(x1, y1), (x2, y2), ...].
    
    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False

    px, py = polygon[0]
    for i in range(1, n + 1):
        qx, qy = polygon[i % n]
        if y > min(py, qy):
            if y <= max(py, qy):
                if x <= max(px, qx):
                    if py != qy:
                        x_intersect = (y - py) * (qx - px) / (qy - py) + px
                    if px == qx or x <= x_intersect:
                        inside = not inside
        px, py = qx, qy

    return inside
