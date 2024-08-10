import numpy as np
from scipy.spatial.distance import cdist

def find_endpoints(curve):
    """
    Find the endpoints of a curve.
    
    :param curve: List of numpy arrays, each representing a segment of the curve
    :return: List of endpoints (start and end points of each segment)
    """
    endpoints = []
    for segment in curve:
        endpoints.append(segment[0])  # Start point
        endpoints.append(segment[-1])  # End point
    return np.array(endpoints)

def find_gaps(curves, threshold=10):
    """
    Find gaps between curves.
    
    :param curves: List of curves, where each curve is a list of numpy arrays
    :param threshold: Maximum distance to consider as a gap
    :return: List of tuples (curve1_index, point1_index, curve2_index, point2_index, distance)
    """
    gaps = []
    for i, curve1 in enumerate(curves):
        endpoints1 = find_endpoints(curve1)
        for j, curve2 in enumerate(curves[i+1:], start=i+1):
            endpoints2 = find_endpoints(curve2)
            distances = cdist(endpoints1, endpoints2)
            close_points = np.argwhere(distances < threshold)
            for p1, p2 in close_points:
                gaps.append((i, p1, j, p2, distances[p1, p2]))
    return sorted(gaps, key=lambda x: x[4])  # Sort by distance

def interpolate_gap(point1, point2, num_points=10):
    """
    Interpolate points between two endpoints to fill a gap.
    
    :param point1: Start point of the gap
    :param point2: End point of the gap
    :param num_points: Number of points to interpolate
    :return: Numpy array of interpolated points
    """
    t = np.linspace(0, 1, num_points)
    return np.outer(1-t, point1) + np.outer(t, point2)

def fill_gaps(curves, threshold=10, max_gaps=None):
    """
    Fill gaps in a set of curves.
    
    :param curves: List of curves, where each curve is a list of numpy arrays
    :param threshold: Maximum distance to consider as a gap
    :param max_gaps: Maximum number of gaps to fill (None for all gaps)
    :return: Updated list of curves with gaps filled
    """
    gaps = find_gaps(curves, threshold)
    if max_gaps is not None:
        gaps = gaps[:max_gaps]
    
    new_curves = [curve.copy() for curve in curves]
    
    for i, p1, j, p2, _ in gaps:
        curve1 = new_curves[i]
        curve2 = new_curves[j]
        
        # Determine which endpoints to connect
        point1 = curve1[-1][-1] if p1 % 2 else curve1[0][0]
        point2 = curve2[-1][-1] if p2 % 2 else curve2[0][0]
        
        # Interpolate gap
        gap_fill = interpolate_gap(point1, point2)
        
        # Add gap fill to appropriate curve
        if p1 % 2:  # If p1 is an end point
            curve1.append(gap_fill)
        else:  # If p1 is a start point
            curve1.insert(0, gap_fill[::-1])
        
        # Merge curves
        if p2 % 2:  # If p2 is an end point
            curve1.extend(curve2)
        else:  # If p2 is a start point
            curve1.extend(curve2[::-1])
        
        # Remove curve2 and update curve1
        new_curves[i] = curve1
        new_curves[j] = []
    
    # Remove empty curves
    new_curves = [curve for curve in new_curves if curve]
    
    return new_curves

# Example usage
if __name__ == "__main__":
    from data_loader import read_csv
    from visualizer import plot_curves
    
    # Load a sample CSV file
    csv_path = "./src/problems/frag0.csv"  # Update this path as needed
    curves = read_csv(csv_path)
    
    # Plot original curves
    plot_curves(curves, "Original Curves")
    
    # Fill gaps
    filled_curves = fill_gaps(curves, threshold=20, max_gaps=5)
    
    # Plot filled curves
    plot_curves(filled_curves, "Curves with Gaps Filled")