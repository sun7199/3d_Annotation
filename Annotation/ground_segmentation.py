import glob
import os
import shutil

import numpy as np
import open3d as o3d
import utils
import pandas
from carBox import get_groundBorder
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import json
from utils import *


def filter_ground_border(points):
    condition = np.logical_and(points[:, 2] < 0.3, points[:, 2] > 0)
    points = points[condition]
    pointClusters = dbscan(points)

    maxcluster = None
    maxClusterLen = 0
    for cluster in pointClusters:
        if len(cluster) > maxClusterLen:
            maxClusterLen = len(cluster)
            maxcluster = cluster

    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 创建窗口，设置窗口标题
    vis.create_window(window_name="point_cloud")
    # 设置点云渲染参数
    opt = vis.get_render_option()
    # 设置背景色（这里为白色）
    opt.background_color = np.array([255, 255, 255])
    # 设置渲染点的大小
    opt.point_size = 3.0

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(maxcluster))
    # 添加点云
    vis.add_geometry(point_cloud)
    vis.run()
    maxcluster=np.array(maxcluster)
    x = maxcluster[:, 0]
    y = maxcluster[:, 1]
    print(x.max(), y.min(), y.max())
    return x.max(), y.min(), y.max()


if __name__ == "__main__":
    pcdPath = "D:/scene_001/002/1688038395.489439.pcd"
    pcd = o3d.io.read_point_cloud(pcdPath)
    # 设置点云颜色 只能是0 1 如[1,0,0]代表红色为既r
    pcd.paint_uniform_color([0, 1, 0])
    points = np.array(pcd.points)
    filter_ground_border(points)
