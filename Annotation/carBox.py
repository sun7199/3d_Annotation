import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import open3d as o3d
import utils


def get_groundBorder(pointCloud, cluster):
    cluster = np.array(cluster)
    x = np.array(cluster[:, 0])
    y = np.array(cluster[:, 1])
    z = []
    for point in pointCloud:
        if x.min() < point[0] < x.max() and y.min() < point[1] < y.max() and point[2] < 0.5:
            z.append(point[2])
    z = np.array(z)
    # Fit linear regression with different slopes (close to zero)
    best_line = 0
    best_line_mse = float('inf')

    for slope in np.linspace(-1, 1, 100):
        z_pred = slope * np.ones_like(z)  # Horizontal line equation
        mse = np.mean((z - z_pred) ** 2)
        if mse < best_line_mse:
            best_line_mse = mse
            best_line = slope
    x_condition = np.logical_and(pointCloud[:, 0] <= x.max(), pointCloud[:, 0] >= x.min())
    y_condition = np.logical_and(pointCloud[:, 1] <= y.max(), pointCloud[:, 1] >= y.min())
    z_condition = pointCloud[:, 2] > best_line

    # Combine conditions
    combined_condition = np.logical_and(np.logical_and(x_condition, y_condition), z_condition)

    # Apply condition to remove elements
    pointCloud = pointCloud[~combined_condition]
    return pointCloud


def get_roadBorder(point_cloud):
    cluster_infos = []
    cluster_info = []
    point_cloud = o3d.io.read_point_cloud(point_cloud)
    points = np.array(point_cloud.points)
    clusters = utils.dbscan(points)
    for cluster in clusters:
        cluster_info.append(len(cluster))
        cluster_info.append(cluster)
    cluster_infos.append(cluster_info)


def fit_box(cluster):
    box_x = 5
    box_y = 2
    box_z = 1.8
    x = cluster[:, 0]
    y = cluster[:, 1]
    z = cluster[:, 2]
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()
    if x_range > 5:
        box_x = x_range
    if y_range > 2:
        box_y = y_range
    if z_range > 1.8:
        box_z = z_range
    return x_range, y_range, z_range


def filter_tooNear(objects):
    pre_point = None
    distance = None
    removeList = []
    for i in range(0, len(objects)):
        for j in range(0, len(objects)):
            if j != i:
                if abs(objects[j][0][0] - objects[i][0][0]) <= 6 and abs(
                        objects[j][0][1] - objects[i][0][1]) <= 2 and abs(objects[j][0][2] - \
                                                                          objects[i][0][2]) <= 1.8:
                    if objects[i][2] < objects[j][2]:
                        removeList.append(i)
                    else:
                        removeList.append(j)
    removeList = list(set(removeList))
    removeList.sort(reverse=True)
    for index in removeList:
        objects.pop(index)
    return objects


def merge_tooNear(objects):
    pre_point = None
    distance = None
    removeList = []
    for i in range(0, len(objects)):
        is_exist = False
        if i in removeList:
            is_exist = True
        if not is_exist:
            for j in range(0, len(objects)):
                if j != i:
                    if abs(objects[j][0][0] - objects[i][0][0]) <= 3 and abs(
                            objects[j][0][1] - objects[i][0][1]) <= 1 and abs(objects[j][0][2] - \
                                                                              objects[i][0][2]) <= 1.8:
                        if objects[i][1][0] > objects[j][1][0]:
                            objects[i][0][0] -= (objects[i][1][0] - objects[j][1][0])

                        if objects[i][1][1] > objects[j][1][1]:
                            objects[i][0][1] -= (objects[i][1][1] - objects[j][1][1])

                        if objects[i][1][2] > objects[j][1][2]:
                            objects[i][0][2] -= (objects[i][1][2] - objects[j][1][2])

                        removeList.append(j)
    removeList = list(set(removeList))
    removeList.sort(reverse=True)
    for index in removeList:
        objects.pop(index)
    return objects


if __name__ == "__main__":
    point_cloud = "003/003/1688782470.889551.pcd"
    get_roadBorder(point_cloud)
