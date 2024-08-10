import numpy as np
import os

def read_csv(csv_path):
    """
    Read a CSV file and parse the data into a nested list structure.
    
    Parameters:
        csv_path (str): The path to the CSV file.
    
    Returns:
        list: A nested list where each element is a list of points (paths).
    """
    try:
        np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    except IOError as e:
        print(f"Error reading {csv_path}: {e}")
        return []
    
    path_XYs = []
    if np_path_XYs.size == 0:
        return path_XYs  # Return an empty list if the file is empty or unreadable
    
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
    Load all CSV files from a specified directory.
    
    Parameters:
        data_dir (str): The directory containing the CSV files.
    
    Returns:
        dict: A dictionary where keys are filenames (without extension) and values are the parsed CSV data.
    """
    dataset = {}
    
    if not os.path.isdir(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return dataset
    
    for filename in os.listdir(data_dir):
        name, ext = os.path.splitext(filename)
        if ext.lower() == '.csv':
            csv_path = os.path.join(data_dir, filename)
            print(f"Loading {csv_path}")
            csv_data = read_csv(csv_path)
            if csv_data:
                dataset[name] = csv_data
            else:
                print(f"Warning: {csv_path} is empty or could not be processed.")
    
    return dataset

# Example usage
if __name__ == "__main__":
    data_dir = "./src/problems"  # Update this path as needed
    dataset = load_dataset(data_dir)
    print(f"Loaded {len(dataset)} CSV files.")
    for name, csv_data in list(dataset.items())[:1]:  # Print details of the first item
        print(f"\nFile: {name}")
        print(f"CSV data: {len(csv_data)} paths")
