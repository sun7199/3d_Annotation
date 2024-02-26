import glob
import os
import shutil

import numpy as np
import open3d as o3d
import utils
import pandas
from Annotation import edge_detection
from carBox import get_groundBorder
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import json

"""
# Function to create a dictionary from input data
# Input data in the specified format
input_data = 
center: 1,2,3
dimension: 5,2,1.8
rotation: 0,-1.5,0
"""


def copy_file(source_path, destination_path):
    """
    Copy a file from source_path to destination_path.

    Parameters:
    - source_path: The path to the source file.
    - destination_path: The path where the file will be copied.
    """
    shutil.copyfile(source_path, destination_path)


def create_dictionary(input_data):
    # Split the input data into lines
    lines = input_data.split('\n')

    # Initialize an empty dictionary
    data = {}

    # Process each line
    for line in lines:
        # Split the line into key and value
        key, values = line.split(':')

        # Split the values into x, y, z components
        components = [float(val.strip()) for val in values.split(',')]

        # Update the dictionary with the key and components
        data[key.strip()] = {"x": components[0], "y": components[1], "z": components[2]}

    return data


def write_json(input_list):
    # Convert the input list to a list of dictionaries
    output_list = input_list

    # Save the list of dictionaries to a JSON file
    json_file_path = "kitty_open3d/output.json"
    with open(json_file_path, "w") as json_file:
        json.dump(output_list, json_file, indent=4)


def dbscan(points):
    epsilon = 0.5
    min_samples = 4
    # Compute DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(points)
    labels = db.labels_

    no_clusters = len(np.unique(labels))
    no_noise = np.sum(np.array(labels) == -1, axis=0)

    clusters = []
    for j in np.unique(labels):
        if j != -1:
            cluster = []
            for i in range(len(labels)):
                if labels[i] == j:
                    cluster.append(points[i])
            clusters.append(cluster)
    return clusters
    # print('Estimated no. of clusters: %d' % no_clusters)
    # print('Estimated no. of noise points: %d' % no_noise)


def threshold_cluster(Data_set, threshold):
    # 统一格式化数据为一维数组
    stand_array = np.asarray(Data_set).ravel('C')
    stand_Data = pandas.Series(stand_array)
    index_list, class_k = [], []
    while stand_Data.any():
        if len(stand_Data) == 1:
            index_list.append(list(stand_Data.index))
            class_k.append(list(stand_Data))
            stand_Data = stand_Data.drop(stand_Data.index)
        else:
            class_data_index = stand_Data.index[0]
            class_data = stand_Data[class_data_index]
            stand_Data = stand_Data.drop(class_data_index)
            if (abs(stand_Data - class_data) <= threshold).any():
                args_data = stand_Data[abs(stand_Data - class_data) <= threshold]
                stand_Data = stand_Data.drop(args_data.index)
                index_list.append([class_data_index] + list(args_data.index))
                class_k.append([class_data] + list(args_data))
            else:
                index_list.append([class_data_index])
                class_k.append([class_data])
    return index_list, class_k


def get_3d_box(center, dimensions, rotation):
    l, w, h = dimensions["length"], dimensions["width"], dimensions["height"]
    corners = np.array([
        [-l / 2, -w / 2, -h / 2], [l / 2, -w / 2, -h / 2], [-l / 2, -w / 2, h / 2], [l / 2, -w / 2, h / 2],
        [-l / 2, w / 2, -h / 2], [l / 2, w / 2, -h / 2], [-l / 2, w / 2, h / 2], [l / 2, w / 2, h / 2]
    ])

    # 转换为弧度
    rx, ry, rz = np.radians(rotation["x"]), np.radians(rotation["y"]), np.radians(rotation["z"])

    # 旋转矩阵
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    R = Rz * Ry * Rx
    corners_rotated = np.dot(corners, R.T)

    # 平移到中心点
    corners_rotated += np.array([center["x"], center["y"], center["z"]])

    return corners_rotated


def compute_3D_box_cam2(dimension, center, yaw):
    '''
    Return:3Xn in cam2 coordinate
    '''
    h, w, l = dimension
    x, y, z = center
    # 建立旋转矩阵R
    R = np.array([ [0, 1, 0],[np.cos(yaw), 0, np.sin(yaw)], [-np.sin(yaw), 0, np.cos(yaw)]])
    # 计算8个顶点坐标
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    # 使用旋转矩阵变换坐标
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # 最后在加上中心点
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2.T


def read_json(filepath):
    with open(filepath, "rb") as file:
        json_data = json.load(file)
    return json_data


def find_border(pointcloud):
    epsilon_x = 0.02
    epsilon_y = 0.01
    min_samples = 300
    border = []
    temp = [pointcloud[0]]
    for i in range(1, len(pointcloud)):
        temp.append(pointcloud[i])
        if (abs(pointcloud[i][0] - pointcloud[i - 1][0]) > epsilon_x or
                abs(pointcloud[i][1] - pointcloud[i - 1][1]) > epsilon_y):
            border.append(np.array(temp))
            temp = []
    return np.array(border)
