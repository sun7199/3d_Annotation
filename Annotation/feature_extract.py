import numpy as np

def compute_hpc(point_cloud, num_cells_x, num_cells_y, num_bins):
    """
    Compute Histogram of Point Clouds (HPC) features.

    Args:
    - point_cloud (np.ndarray): 2D array containing the (x, y) coordinates of points in the point cloud.
    - num_cells_x (int): Number of cells in the x-direction.
    - num_cells_y (int): Number of cells in the y-direction.
    - num_bins (int): Number of bins for the histogram in each cell.

    Returns:
    - hpc_features (np.ndarray): Flattened array of HPC features.
    """
    # Compute cell width and height
    cell_width = (np.max(point_cloud[:, 0]) - np.min(point_cloud[:, 0])) / num_cells_x
    cell_height = (np.max(point_cloud[:, 1]) - np.min(point_cloud[:, 1])) / num_cells_y

    # Initialize HPC features array
    hpc_features = np.zeros((num_cells_x, num_cells_y, num_bins))

    # Iterate over cells
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            # Define cell boundaries
            x_min = np.min(point_cloud[:, 0]) + i * cell_width
            x_max = x_min + cell_width
            y_min = np.min(point_cloud[:, 1]) + j * cell_height
            y_max = y_min + cell_height

            # Extract points within the cell
            cell_points = point_cloud[(point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] < x_max) &
                                      (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] < y_max)]

            # Compute histogram of point counts in the cell
            hist, _ = np.histogramdd(cell_points, bins=num_bins, range=[(x_min, x_max), (y_min, y_max)])

            # Normalize histogram
            hist = hist / np.sum(hist)

            # Store histogram in HPC features array
            hpc_features[i, j] = hist.flatten()

    # Flatten HPC features array
    hpc_features = hpc_features.flatten()

    return hpc_features

# Example usage:
if __name__ == "__main__":
    # Assuming 'point_cloud' is a 2D numpy array containing the (x, y) coordinates of points in the point cloud
    point_cloud = np.random.rand(100, 2)  # Example point cloud with 100 points
    num_cells_x = 5
    num_cells_y = 5
    num_bins = 10

    hpc_features = compute_hpc(point_cloud, num_cells_x, num_cells_y, num_bins)
    print("HPC Features:", hpc_features)
