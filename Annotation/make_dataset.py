import open3d as o3d
import json
from utils import *


def pick_points(pcd, jsonPath):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)

    jsonfile = read_json(jsonPath)
    for obj in jsonfile["objects"]:
        # 解析每个目标的数据
        center = obj["box3d"]["center"]
        dimensions = obj["box3d"]["dimensions"]
        rotation = obj["box3d"]["rotation"]
        box_corners = get_3d_box(center, dimensions, rotation)

        # two different box points sequence
        # lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
        #                   [0, 4], [1, 5], [2, 6], [3, 7]])
        lines_box = np.array([[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                              [0, 4], [1, 5], [2, 6], [3, 7]])
        # 设置点与点之间线段的颜色
        colors = np.array([[1, 0, 0] for j in range(len(lines_box))])
        # 创建Bbox候选框对象
        line_set = o3d.geometry.LineSet()
        # 将八个顶点连接次序的信息转换成o3d可以使用的数据类型
        line_set.lines = o3d.utility.Vector2iVector(lines_box)
        # 设置每条线段的颜色
        line_set.colors = o3d.utility.Vector3dVector(colors)
        # 把八个顶点的空间信息转换成o3d可以使用的数据类型
        line_set.points = o3d.utility.Vector3dVector(box_corners)
        print(line_set.points)
        # 将矩形框加入到窗口中
        vis.add_geometry(line_set)

    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def collect_points(cloud_points, picked_points):
    jsonfile = read_json(jsonPath)
    for obj in jsonfile["objects"]:
        # 解析每个目标的数据
        center = obj["box3d"]["center"]
        dimensions = obj["box3d"]["dimensions"]
        rotation = obj["box3d"]["rotation"]
        box_corners = get_3d_box(center, dimensions, rotation)
        x_min = np.min(box_corners[:, 0])
        x_max = np.min(box_corners[:, 0])
        y_min = np.min(box_corners[:, 1])
        y_max = np.min(box_corners[:, 1])
        z_min = np.min(box_corners[:, 2])
        z_max = np.min(box_corners[:, 2])

    cluster = []
    clusters = []

    for i in picked_points:
        for j in cloud_points:
            if x_min <= j[0] <= x_max and y_min <= j[1] <= y_max and z_min <= j[2] <= z_max:
                cluster.append(j)
        clusters.append(cluster)

    return clusters


if __name__ == "__main__":
    pcdPath = "D:/003/003/1688782470.889551.pcd"
    jsonPath = "D:/003/003/1688782470.889551.json"
    point_cloud = o3d.io.read_point_cloud(pcdPath)
    cloud_points = np.array(point_cloud.points)
    picked_points = pick_points(point_cloud, jsonPath)
