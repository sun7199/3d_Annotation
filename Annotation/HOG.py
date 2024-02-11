import open3d as o3d
import numpy as np
from skimage.feature import hog
from skimage import exposure


def compute_3d_hog(point_cloud, voxel_size=2):
    # Compute gradients in x, y, and z directions
    gradients_x = np.gradient(point_cloud[:, 0])
    gradients_y = np.gradient(point_cloud[:, 1])
    gradients_z = np.gradient(point_cloud[:, 2])

    # Compute gradient magnitude and angle
    gradient_magnitude = np.sqrt(gradients_x ** 2 + gradients_y ** 2 + gradients_z ** 2)
    gradient_angle = np.arctan2(gradients_y, gradients_x)

    # Novelization
    voxel_indices = np.floor(point_cloud[:, :3] / voxel_size).astype(int)
    voxel_gradients = np.vstack((gradients_x, gradients_y, gradients_z, gradient_magnitude, gradient_angle)).T

    # Compute histograms for each voxel
    num_bins = 9
    histograms = []

    for voxel_index in np.unique(voxel_indices, axis=0):
        mask = np.all(voxel_indices == voxel_index, axis=1)
        voxel_histogram, _ = np.histogramdd(voxel_gradients[mask],
                                            bins=[num_bins, num_bins, num_bins, num_bins, num_bins],
                                            range=[(-1, 1), (-1, 1), (-1, 1), (0, np.max(gradient_magnitude)),
                                                   (-np.pi, np.pi)])
        histograms.append(voxel_histogram.flatten())

    # Global feature vector
    global_feature = np.concatenate(histograms)

    return global_feature


# Example usage
# Assuming 'point_cloud' is your 3D point cloud represented as a NumPy array
# Each row of 'point_cloud' should contain the (x, y, z) coordinates of a point

# result = compute_3d_hog_with_magnitude_and_angle(point_cloud)


def generate_3d_box_coordinates(center, dimensions, unit):
    """
    Generate coordinates of all points within a 3D box.

    Parameters:
    - center: Tuple or list containing (x, y, z) coordinates of the box center.
    - dimensions: Tuple or list containing (length, width, height) of the box.

    Returns:
    - coordinates: Numpy array containing the coordinates of all points within the box.
    """
    length, width, height = dimensions
    min_x, min_y, min_z = center[0] - (length * unit) / 2, center[1] - (width * unit) / 2, center[2] - (
            height * unit) / 2
    max_x, max_y, max_z = center[0] + (length * unit) / 2, center[1] + (width * unit) / 2, center[2] + (
            height * unit) / 2

    # Generate coordinates within the box
    x_coords = np.linspace(min_x, max_x, num=int(length + 1))
    y_coords = np.linspace(min_y, max_y, num=int(width + 1))
    z_coords = np.linspace(min_z, max_z, num=int(height + 1))

    # Create a meshgrid of coordinates
    x, y, z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # Stack the coordinates into a (N, 3) array
    coordinates = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    return coordinates


def getFeature(center, dimensions, unit):
    # Round center point to two decimal places
    center = np.array(list(center.values()))
    center = np.round(center, 1)
    # create box and get coordinates
    box_coordinates = generate_3d_box_coordinates(center, dimensions, unit)
    # get feature map of this box
    result = compute_3d_hog(box_coordinates)
    return result


def compareFeature(feature1, feature2):
    # Count the number of equal values
    feature1=np.array(feature1)
    feature2=np.array(feature2)
    equal_values_count = np.sum(feature1 == feature2)
    return equal_values_count

# for debug
if __name__ == "__main__":
    center_point1 = (1.3232, 4.234, 7.4234232)
    center_point2 = (3.1231, 5.3422, 9.4342)
    box_dimensions = (50, 20, 18)
    unit = 0.1
    feature1=getFeature(center_point1,box_dimensions,unit)
    feature2 = getFeature(center_point2, box_dimensions, unit)
    result=compareFeature(feature1,feature2)
    print(result)