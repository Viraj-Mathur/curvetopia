import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def plot_paths(paths, ax=None, color='blue', linewidth=2):
    """
    Plot a series of paths (curves) on a given axis.
    
    Parameters:
        paths (list): A list of paths, where each path is a list of (x, y) points.
        ax (matplotlib.axes._axes.Axes): The axis to plot on. If None, creates a new figure and axis.
        color (str): Color of the paths.
        linewidth (int): Width of the paths.
    
    Returns:
        matplotlib.axes._axes.Axes: The axis with the plotted paths.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    for path in paths:
        for segment in path:
            segment = np.array(segment)
            ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=linewidth)
    
    ax.set_aspect('equal', 'box')
    ax.invert_yaxis()  # To match the SVG coordinate system
    return ax

def save_plot_as_svg(fig, output_path):
    """
    Save a matplotlib figure as an SVG file.
    
    Parameters:
        fig (matplotlib.figure.Figure): The figure to save.
        output_path (str): The path where the SVG should be saved.
    """
    fig.savefig(output_path, format='svg')
    print(f"Saved SVG to {output_path}")

def save_plot_as_png(fig, output_path):
    """
    Save a matplotlib figure as a PNG file.
    
    Parameters:
        fig (matplotlib.figure.Figure): The figure to save.
        output_path (str): The path where the PNG should be saved.
    """
    fig.savefig(output_path, format='png')
    print(f"Saved PNG to {output_path}")

def visualize_and_save(paths, output_dir, filename):
    """
    Visualize paths and save the plot as both SVG and PNG.
    
    Parameters:
        paths (list): A list of paths to visualize.
        output_dir (str): The directory to save the SVG and PNG files.
        filename (str): The base filename (without extension) for the saved files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig, ax = plt.subplots()
    ax = plot_paths(paths, ax=ax)
    
    svg_path = os.path.join(output_dir, f"{filename}.svg")
    png_path = os.path.join(output_dir, f"{filename}.png")
    
    save_plot_as_svg(fig, svg_path)
    save_plot_as_png(fig, png_path)
    
    plt.close(fig)

def visualize_polygon(vertices, ax=None, color='green', linewidth=2, fill=False):
    """
    Visualize a polygon given its vertices.
    
    Parameters:
        vertices (list of tuples): A list of (x, y) tuples representing the vertices of the polygon.
        ax (matplotlib.axes._axes.Axes): The axis to plot on. If None, creates a new figure and axis.
        color (str): Color of the polygon edges.
        linewidth (int): Width of the polygon edges.
        fill (bool): Whether to fill the polygon with color.
    
    Returns:
        matplotlib.axes._axes.Axes: The axis with the plotted polygon.
    """
    if ax is None:
        fig, ax = plt.subplots()

    polygon = patches.Polygon(vertices, closed=True, edgecolor=color, facecolor=color if fill else 'none', linewidth=linewidth)
    ax.add_patch(polygon)
    
    ax.set_aspect('equal', 'box')
    ax.invert_yaxis()  # To match the SVG coordinate system
    return ax

# Example usage
if __name__ == "__main__":
    example_paths = [
        [[[0, 0], [1, 1], [2, 0]], [[3, 3], [4, 4], [5, 3]]],  # Example set of paths
        [[[1, 2], [2, 3], [3, 2]]]
    ]
    output_dir = "./src/output"  # Update this path as needed
    visualize_and_save(example_paths, output_dir, "example_plot")
