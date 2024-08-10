import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def detect_lines(paths_XYs, threshold=0.01):
    """
    Detects straight lines from given paths (polylines).
    
    Parameters:
        paths_XYs (list): List of paths, where each path is a numpy array of points.
        threshold (float): Error threshold to determine if a polyline can be approximated as a straight line.

    Returns:
        line_segments (list): List of detected line segments. Each line is represented by two points.
    """
    line_segments = []

    for path in paths_XYs:
        for XY in path:
            if XY.shape[0] < 2:  # Less than 2 points cannot form a line
                continue
            
            # Perform linear regression to fit a line
            model = LinearRegression()
            X = XY[:, 0].reshape(-1, 1)
            y = XY[:, 1]
            model.fit(X, y)
            predictions = model.predict(X)
            
            # Calculate the mean squared error to check if it's a straight line
            error = mean_squared_error(y, predictions)
            if error < threshold:
                line_segments.append([XY[0], XY[-1]])  # Store as line segment (start and end points)

    return line_segments

if __name__ == "__main__":
    # Example usage
    from src.data_loader import read_csv
    csv_path = "./src/problems"  # Update this path as needed
    curves = read_csv(csv_path)
    
    detected_lines = detect_lines(curves)
    
    # Print detected line segments
    for line in detected_lines:
        print(f"Line segment from {line[0]} to {line[1]}")
