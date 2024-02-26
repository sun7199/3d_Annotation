import numpy as np
from scipy.spatial import Delaunay
import open3d as o3d
import matplotlib.pyplot as plt
import alphashape
import warnings


def find_roadBorder(points):
    condition = np.logical_and(points[:, 2] < 0.3, points[:, 2] > 0)
    points = points[condition]

    points = points[:, :2]

    # Set the alpha value (adjust as needed)
    alpha_value = 0.1

    # Compute the alpha shape
    alpha_shape = alphashape.alphashape(points, alpha_value)

    # Get the alpha shape boundary
    boundary = np.array(alpha_shape.exterior.coords)

    boundary = np.round(boundary, decimals=2)
    temp=[]
    up_boarder=[]
    down_boarder=[]
    x_min = np.min(boundary[:, 0])
    y_min = np.min(boundary[:, 1])
    x_max = np.max(boundary[:, 0])
    y_max = np.max(boundary[:, 1])
    temp = boundary[boundary[:,1]>y_max-2]
    for i in range(0, len(temp)):
        up_boarder.append(np.append(temp[i],0))
    temp = boundary[boundary[:, 1] < y_min + 2]
    for i in range(0, len(temp)):
        down_boarder.append(np.append(temp[i], 0))
    up_boarder=np.array(up_boarder)
    down_boarder = np.array(down_boarder)
    return up_boarder, down_boarder

if __name__=="__main__":
    pcdfile = '003/003/1688782470.889551.pcd'
    pcd = o3d.io.read_point_cloud(pcdfile)
    points = np.array(pcd.points)
    ground_points = points

    up_boarder, down_boarder = find_roadBorder(points)
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

    vis.add_geometry(pcd)

    # Create Open3D lines
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(up_boarder)
    # Generate lines by connecting consecutive points
    lines = [[i, i + 1] for i in range(len(up_boarder) - 1)]
    line_set.lines = o3d.utility.Vector2iVector(lines)
    # Set the color of the lines
    line_color = [1, 0, 0]  # Red color
    num_lines = len(line_set.lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(line_color, (num_lines, 1)))

    vis.add_geometry(line_set)

    # Create Open3D lines
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(down_boarder)
    # Generate lines by connecting consecutive points
    lines = [[i, i + 1] for i in range(len(down_boarder) - 1)]
    line_set.lines = o3d.utility.Vector2iVector(lines)
    # Set the color of the lines
    line_color = [1, 0, 0]  # Red color
    num_lines = len(line_set.lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(line_color, (num_lines, 1)))

    vis.add_geometry(line_set)

    vis.run()
