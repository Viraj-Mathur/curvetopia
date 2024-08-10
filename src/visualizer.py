import matplotlib.pyplot as plt
import numpy as np
import svgwrite
import cairosvg

def plot_curves(paths_XYs, title="Curve Visualization"):
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define a list of colors
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.title(title)
    plt.show()

def save_as_svg(paths_XYs, filename):
    """
    Save the paths as an SVG file.
    """
    dwg = svgwrite.Drawing(filename, profile='tiny')
    for path in paths_XYs:
        for segment in path:
            points = [tuple(point) for point in segment]
            dwg.add(dwg.polyline(points=points, fill='none', stroke='black'))
    dwg.save()

def save_as_png(paths_XYs, svg_path):
    """
    Save the paths as an SVG file and convert it to a PNG.
    """
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))

    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)

    # Create a new SVG drawing
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()

    colours = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#33FFF3']  # Add more colors as needed

    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path:
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
        group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))

    dwg.add(group)
    dwg.save()

    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 // min(H, W))
    cairosvg.svg2png(url=svg_path, write_to=png_path, parent_width=W, parent_height=H, output_width=fact * W, output_height=fact * H, background_color='white')
    return png_path

# Example usage
if __name__ == "__main__":
    from data_loader import load_dataset

    data_dir = "./src/problems"  # Update this path as needed
    dataset = load_dataset(data_dir)

    for name, csv_data in list(dataset.items())[:1]:  # Visualize the first item
        plot_curves(csv_data, f"CSV Data: {name}")

        # Save CSV data as SVG
        svg_filename = f"{name}_from_csv.svg"
        save_as_svg(csv_data, svg_filename)
        print(f"Saved {svg_filename}")

        # Save SVG as PNG
        png_filename = save_as_png(csv_data, svg_filename)
        print(f"Saved {png_filename}")
