import os
from src.data_loader import load_dataset
from src.visualizer import plot_curves, save_as_svg, save_as_png

def main():
    data_dir = "./src/problems"  # Update this path to where your CSV and SVG files are located
    dataset = load_dataset(data_dir)

    if not dataset:
        print("No CSV files found in the specified directory.")
        return

    print(f"Found {len(dataset)} CSV files.")

    svg_output_dir = "./output"
    os.makedirs(svg_output_dir, exist_ok=True)

    for name, csv_data in dataset.items():
        try:
            print(f"Processing {name}")
            
            # Plot original data
            plot_curves(csv_data, f"Original: {name}")
            
            # Save original data as SVG
            svg_filename = os.path.join(svg_output_dir, f"{name}_from_csv.svg")
            save_as_svg(csv_data, svg_filename)
            
            # Convert SVG to PNG
            png_filename = save_as_png(csv_data, svg_filename)
            print(f"Saved {svg_filename} and {png_filename}")
            
            print(f"Finished processing {name}")
        except Exception as e:
            print(f"Error processing {name}: {e}")
        print("------------------------")

if __name__ == "__main__":
    main()
