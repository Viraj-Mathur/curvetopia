import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def hough_circle(points, radii_range, threshold=0.5):
    """
    Detect circles using Hough Transform.
    
    :param points: numpy array of shape (n, 2) containing the points
    :param radii_range: tuple (min_radius, max_radius)
    :param threshold: accumulator threshold for circle detection
    :return: list of tuples (x, y, r) for detected circles
    """
    min_radius, max_radius = radii_range
    x, y = points[:, 0], points[:, 1]
    
    # Create accumulator
    accumulator = {}
    
    # Loop through all points and possible radii
    for i in range(len(x)):
        for r in range(min_radius, max_radius + 1):
            for t in range(0, 360, 5):  # Step of 5 degrees
                a = x[i] - r * np.cos(np.radians(t))
                b = y[i] - r * np.sin(np.radians(t))
                a, b = int(a), int(b)
                if (a, b, r) in accumulator:
                    accumulator[(a, b, r)] += 1
                else:
                    accumulator[(a, b, r)] = 1
    
    # Find peaks in accumulator
    circles = [k for k, v in accumulator.items() if v > threshold * len(points)]
    
    return circles

def fit_circle_optimize(points):
    """
    Fit a circle to points using least squares optimization.
    
    :param points: numpy array of shape (n, 2) containing the points
    :return: tuple (x, y, r) for the best-fit circle
    """
    def calc_R(xc, yc):
        return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)

    def f_2(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = np.mean(points, axis=0)
    center, _ = optimize.leastsq(f_2, center_estimate)
    
    radius = np.mean(calc_R(*center))
    return (*center, radius)

def detect_circles(points, min_radius=10, max_radius=100, min_points=5):
    """
    Detect and fit circles in a set of points.
    
    :param points: numpy array of shape (n, 2) containing all points
    :param min_radius: minimum radius to consider
    :param max_radius: maximum radius to consider
    :param min_points: minimum number of points to constitute a circle
    :return: list of tuples (x, y, r) for detected circles
    """
    # Initial circle detection using Hough Transform
    initial_circles = hough_circle(points, (min_radius, max_radius))
    
    # Cluster detected circles
    circle_params = np.array(initial_circles)
    clustering = DBSCAN(eps=max_radius/2, min_samples=2).fit(circle_params)
    
    refined_circles = []
    for label in set(clustering.labels_):
        if label == -1:  # Noise points
            continue
        
        # Get average circle parameters for this cluster
        cluster_circles = circle_params[clustering.labels_ == label]
        avg_circle = np.mean(cluster_circles, axis=0)
        
        # Find points close to this circle
        distances = np.abs(np.sqrt((points[:, 0] - avg_circle[0])**2 + 
                                   (points[:, 1] - avg_circle[1])**2) - avg_circle[2])
        circle_points = points[distances < max_radius/10]
        
        if len(circle_points) >= min_points:
            # Refine circle fit
            x, y, r = fit_circle_optimize(circle_points)
            refined_circles.append((x, y, r))
    
    return refined_circles

def plot_circles(points, circles):
    """
    Plot points and detected circles.
    
    :param points: numpy array of shape (n, 2) containing all points
    :param circles: list of tuples (x, y, r) for detected circles
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], s=1)
    
    for x, y, r in circles:
        circle = plt.Circle((x, y), r, fill=False, color='r')
        plt.gca().add_artist(circle)
    
    plt.axis('equal')
    plt.title("Detected Circles")
    plt.show()

# Example usage
if __name__ == "__main__":
    from data_loader import read_csv
    
    # Load a sample CSV file
    csv_path = "./src/problems"  # Update this path as needed
    curves = read_csv(csv_path)
    
    # Combine all points from all curves
    all_points = np.vstack(curves)
    
    # Detect circles
    detected_circles = detect_circles(all_points)
    
    # Plot results
    plot_circles(all_points, detected_circles)
    
    print(f"Detected {len(detected_circles)} circles:")
    for i, (x, y, r) in enumerate(detected_circles, 1):
        print(f"Circle {i}: center = ({x:.2f}, {y:.2f}), radius = {r:.2f}")