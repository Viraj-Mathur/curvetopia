import numpy as np
from scipy import linalg

def fit_ellipse(x, y):
    """
    Fit an ellipse to the given set of points using the least squares method.
    
    Args:
    x, y: Arrays of x and y coordinates of the points
    
    Returns:
    center_x, center_y, a, b, angle: Parameters of the fitted ellipse
    """
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = linalg.eig(np.dot(linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    
    # Extract ellipse parameters
    b, c, d, f, g, a = a[1]/2., a[2], a[3]/2., a[4]/2., a[5], a[0]
    num = b*b - a*c
    center_x = (c*d - b*f) / num
    center_y = (a*f - b*d) / num
    
    up = 2 * (a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
    down1 = (b*b - a*c) * ((c-a) * np.sqrt(1 + 4*b*b/((a-c)*(a-c))) - (c+a))
    down2 = (b*b - a*c) * ((a-c) * np.sqrt(1 + 4*b*b/((a-c)*(a-c))) - (c+a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    
    # Semi-major and semi-minor axes
    a, b = max(res1, res2), min(res1, res2)
    
    # Angle of rotation
    angle = 0.5 * np.arctan(2*b / (a-c))
    
    return center_x, center_y, a, b, angle

def detect_ellipse(points, tolerance=0.1):
    """
    Detect if the given points form an ellipse.
    
    Args:
    points: numpy array of shape (n, 2) containing x, y coordinates
    tolerance: maximum allowed average distance from points to the fitted ellipse
    
    Returns:
    is_ellipse: boolean indicating if the points form an ellipse
    params: parameters of the fitted ellipse (center_x, center_y, a, b, angle) if is_ellipse is True, else None
    """
    x, y = points[:, 0], points[:, 1]
    
    # Fit ellipse
    center_x, center_y, a, b, angle = fit_ellipse(x, y)
    
    # Calculate distances from points to the fitted ellipse
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    x_centered = x - center_x
    y_centered = y - center_y
    x_rot = x_centered * cos_angle + y_centered * sin_angle
    y_rot = -x_centered * sin_angle + y_centered * cos_angle
    distances = np.abs(x_rot**2 / a**2 + y_rot**2 / b**2 - 1)
    
    # Check if the average distance is within the tolerance
    is_ellipse = np.mean(distances) < tolerance
    
    return is_ellipse, (center_x, center_y, a, b, angle) if is_ellipse else None

def detect_ellipses_in_curves(curves, tolerance=0.1):
    """
    Detect ellipses in the given set of curves.
    
    Args:
    curves: List of curves, where each curve is a list of numpy arrays
    tolerance: Maximum allowed average distance from points to the fitted ellipse
    
    Returns:
    List of tuples (is_ellipse, params) for each curve
    """
    results = []
    for curve in curves:
        # Concatenate all points in the curve
        points = np.concatenate(curve)
        is_ellipse, params = detect_ellipse(points, tolerance)
        results.append((is_ellipse, params))
    return results

# Example usage
if __name__ == "__main__":
    import sys
    sys.path.append('./src')  # Add the src directory to the Python path
    from data_loader import read_csv  # Import the read_csv function

    csv_path = "./src/problems.csv"  # Update this path as needed
    curves = read_csv(csv_path)
    
    results = detect_ellipses_in_curves(curves)
    
    for i, (is_ellipse, params) in enumerate(results):
        if is_ellipse:
            print(f"Ellipse detected in curve {i}!")
            print(f"Center: ({params[0]:.2f}, {params[1]:.2f})")
            print(f"Semi-major axis: {params[2]:.2f}")
            print(f"Semi-minor axis: {params[3]:.2f}")
            print(f"Angle: {params[4]:.2f} radians")
        else:
            print(f"No ellipse detected in curve {i}.")
        print()