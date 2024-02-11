import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


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
