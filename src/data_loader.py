import numpy as np
import os

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def load_dataset(data_dir):
    """
    Load all CSV files from a directory.
    Returns a dictionary where keys are filenames (without extension) and values are the csv_data.
    """
    dataset = {}
    for filename in os.listdir(data_dir):
        name, ext = os.path.splitext(filename)
        if ext.lower() == '.csv':
            csv_path = os.path.join(data_dir, filename)
            print(f"Loading {csv_path}")
            csv_data = read_csv(csv_path)
            dataset[name] = csv_data
    return dataset

# Example usage
if __name__ == "__main__":
    data_dir = "./src/problems"  # Update this path as needed
    dataset = load_dataset(data_dir)
    print(f"Loaded {len(dataset)} CSV files.")
    for name, csv_data in list(dataset.items())[:1]:  # Print details of the first item
        print(f"\nFile: {name}")
        print(f"CSV data: {len(csv_data)} paths")