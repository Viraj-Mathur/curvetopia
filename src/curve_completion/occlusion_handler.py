import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def fit_circle(points):
    """
    Fit a circle to a set of 2D points using least squares optimization.
    
    :param points: numpy array of shape (n, 2) containing the points
    :return: tuple (center_x, center_y, radius)
    """
    def calc_R(xc, yc):
        return np.sqrt((points[:,0]-xc)**2 + (points[:,1]-yc)**2)

    def f(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(points, axis=0)
    center, _ = minimize(lambda c: np.sum(f(c)**2), center_estimate).x

    radius = np.mean(calc_R(*center))
    return (*center, radius)

def complete_circle(points, num_points=100):
    """
    Complete a partial circle.
    
    :param points: numpy array of shape (n, 2) containing the partial circle points
    :param num_points: number of points to generate for the complete circle
    :return: numpy array of shape (num_points, 2) representing the completed circle
    """
    center_x, center_y, radius = fit_circle(points)
    theta = np.linspace(0, 2*np.pi, num_points)
    completed_circle = np.column_stack([
        center_x + radius * np.cos(theta),
        center_y + radius * np.sin(theta)
    ])
    return completed_circle

def fit_rectangle(points):
    """
    Fit a rectangle to a set of 2D points.
    
    :param points: numpy array of shape (n, 2) containing the points
    :return: tuple (center_x, center_y, width, height, angle)
    """
    # Compute the oriented bounding box
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    # Rotate points
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    rotated = centered.dot(rot)

    # Get axis-aligned bounding box
    min_coords = np.min(rotated, axis=0)
    max_coords = np.max(rotated, axis=0)
    width = max_coords[0] - min_coords[0]
    height = max_coords[1] - min_coords[1]

    return (*mean, width, height, angle)

def complete_rectangle(points, num_points=100):
    """
    Complete a partial rectangle.
    
    :param points: numpy array of shape (n, 2) containing the partial rectangle points
    :param num_points: number of points to generate for each side of the rectangle
    :return: numpy array of shape (4*num_points, 2) representing the completed rectangle
    """
    center_x, center_y, width, height, angle = fit_rectangle(points)
    
    # Generate rectangle points
    t = np.linspace(0, 1, num_points)
    side1 = np.column_stack([-width/2 + t*width, np.full(num_points, -height/2)])
    side2 = np.column_stack([np.full(num_points, width/2), -height/2 + t*height])
    side3 = np.column_stack([width/2 - t*width, np.full(num_points, height/2)])
    side4 = np.column_stack([np.full(num_points, -width/2), height/2 - t*height])
    
    rectangle = np.vstack([side1, side2, side3, side4])
    
    # Rotate and translate
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    completed_rectangle = rectangle.dot(rot) + [center_x, center_y]
    
    return completed_rectangle

def identify_shape(points):
    """
    Identify the shape of a set of points (circle or rectangle).
    
    :param points: numpy array of shape (n, 2) containing the points
    :return: string 'circle' or 'rectangle'
    """
    _, _, radius = fit_circle(points)
    center_x, center_y, width, height, _ = fit_rectangle(points)
    
    circle_error = np.mean(np.abs(np.sqrt((points[:,0]-center_x)**2 + (points[:,1]-center_y)**2) - radius))
    rectangle_error = np.mean(np.minimum(np.abs(points[:,0] - (center_x - width/2)), np.abs(points[:,0] - (center_x + width/2))) +
                              np.minimum(np.abs(points[:,1] - (center_y - height/2)), np.abs(points[:,1] - (center_y + height/2))))
    
    return 'circle' if circle_error < rectangle_error else 'rectangle'

def handle_occlusions(curves, eps=5, min_samples=5):
    """
    Handle occlusions in a set of curves.
    
    :param curves: List of curves, where each curve is a numpy array of shape (n, 2)
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point
    :return: List of completed curves
    """
    # Flatten all points
    all_points = np.vstack(curves)
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(all_points)
    
    completed_curves = []
    for label in set(clustering.labels_):
        if label == -1:  # Noise points
            continue
        
        cluster_points = all_points[clustering.labels_ == label]
        shape = identify_shape(cluster_points)
        
        if shape == 'circle':
            completed_curve = complete_circle(cluster_points)
        elif shape == 'rectangle':
            completed_curve = complete_rectangle(cluster_points)
        
        completed_curves.append(completed_curve)
    
    return completed_curves

# Example usage
if __name__ == "__main__":
    from data_loader import read_csv
    from visualizer import plot_curves
    
    # Load a sample CSV file
    csv_path = "./src/problems/occlusion2.csv"  # Update this path as needed
    curves = read_csv(csv_path)
    
    # Plot original curves
    plot_curves(curves, "Original Curves with Occlusions")
    
    # Handle occlusions
    completed_curves = handle_occlusions(curves)
    
    # Plot completed curves
    plt.figure(figsize=(10, 10))
    for curve in completed_curves:
        plt.plot(curve[:, 0], curve[:, 1])
    plt.title("Completed Curves")
    plt.axis('equal')
    plt.show()