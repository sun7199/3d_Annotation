import numpy as np
import open3d as o3d

def boundary_extraction(cloud, resolution):
    px_min = np.min(cloud, axis=0)
    px_max = np.max(cloud, axis=0)

    delta_x = (px_max[0] - px_min[0]) / resolution
    minmax_x = np.full((resolution + 1, 2), [np.inf, -np.inf])
    indexs_x = np.zeros(2 * resolution + 2, dtype=int)

    for i, point in enumerate(cloud):
        id = int((point[0] - px_min[0]) / delta_x)
        if point[1] < minmax_x[id, 0]:
            minmax_x[id, 0] = point[1]
            indexs_x[id] = i
        elif point[1] > minmax_x[id, 1]:
            minmax_x[id, 1] = point[1]
            indexs_x[id + resolution + 1] = i

    py_min = np.min(cloud, axis=0)
    py_max = np.max(cloud, axis=0)

    delta_y = (py_max[1] - py_min[1]) / resolution
    minmax_y = np.full((resolution + 1, 2), [np.inf, -np.inf])
    indexs_y = np.zeros(2 * resolution + 2, dtype=int)

    for i, point in enumerate(cloud):
        id = int((point[1] - py_min[1]) / delta_y)
        if point[0] < minmax_y[id, 0]:
            minmax_y[id, 0] = point[0]
            indexs_y[id] = i
        elif point[0] > minmax_y[id, 1]:
            minmax_y[id, 1] = point[0]
            indexs_y[id + resolution + 1] = i

    cloud_xboundary = cloud[indexs_x]
    cloud_yboundary = cloud[indexs_y]

    cloud_boundary = np.concatenate((cloud_xboundary, cloud_yboundary), axis=0)

    return cloud_boundary

if __name__ == "__main__":
    pcdfile = '003/003/1688782470.889551.pcd'
    pcd = o3d.io.read_point_cloud(pcdfile)
    points = np.array(pcd.points)
    condition = np.logical_and(points[:, 2] < 0.3, points[:, 2] > 0)
    points = points[condition]
    ground_points = points
    points = points[:, :2]
    # Define resolution
    resolution = 200

    # Extract boundary
    cloud_boundary = boundary_extraction(points, resolution)
    cloud_boundary = np.hstack((cloud_boundary, np.ones((cloud_boundary.shape[0], 1))))

    # Create Open3D lines
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(cloud_boundary)
    # Generate lines by connecting consecutive points
    lines = [[i, i + 1] for i in range(len(cloud_boundary) - 1)]
    line_set.lines = o3d.utility.Vector2iVector(lines)

    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(ground_points)

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

    vis.add_geometry(line_set)
    vis.add_geometry(pcd)
    vis.run()
