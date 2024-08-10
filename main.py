import os
from src.data_loader import load_dataset
from src.visualizer import plot_paths, save_plot_as_svg, save_plot_as_png

# Constants for directories
DATA_DIR = "./src/problems"  # Update this path to where your CSV and SVG files are located
OUTPUT_DIR = "./output"

def process_file(name, csv_data, output_dir):
    """
    Process a single CSV file: plot, save as SVG, and convert to PNG.
    
    Parameters:
        name (str): The name of the CSV file (without extension).
        csv_data (list): The parsed data from the CSV file.
        output_dir (str): The directory to save the output SVG and PNG files.
    """
    print(f"Processing {name}")

    # Plot original data
    fig, ax = plot_paths(csv_data, color='blue')
    
    # Save original data as SVG
    svg_filename = os.path.join(output_dir, f"{name}_from_csv.svg")
    save_plot_as_svg(fig, svg_filename)
    
    # Save as PNG
    png_filename = os.path.join(output_dir, f"{name}_from_csv.png")
    save_plot_as_png(fig, png_filename)
    
    print(f"Saved {svg_filename} and {png_filename}")
    print(f"Finished processing {name}")

def main():
    # Load the dataset
    dataset = load_dataset(DATA_DIR)
    
    if not dataset:
        print("No CSV files found in the specified directory.")
        return
    
    print(f"Found {len(dataset)} CSV files.")
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each file in the dataset
    for name, csv_data in dataset.items():
        try:
            process_file(name, csv_data, OUTPUT_DIR)
        except Exception as e:
            print(f"Error processing {name}: {e}")
        print("------------------------")

if __name__ == "__main__":
    main()
