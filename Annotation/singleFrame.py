import glob
import os
import shutil

import numpy as np
import open3d as o3d
import utils
import pandas
from Annotation import edge_detection, carBox
from carBox import get_groundBorder
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import json
import roadBoarder
from utils import *
import ground_segmentation


def singleFrame(pcdPath, jsonPath, folder_path):
    pcd = o3d.io.read_point_cloud(pcdPath)
    # 设置点云颜色 只能是0 1 如[1,0,0]代表红色为既r
    pcd.paint_uniform_color([0, 1, 0])
    points = np.array(pcd.points)
    # ground_points = ground_segmentation.filter_ground_border(points)
    border_points, filter_up, filter_down, filter_right = roadBoarder.get_roadBoarder(points)

    # sorted according to y, x,z sequence
    sortedindex = np.lexsort((points[:, 2], points[:, 0], points[:, 1]))
    points = points[sortedindex, :]
    origin_points = points
    # refined_left_edge_y,refined_right_edge_y=edge_detection.getEdge(pcdPath)
    condition1 = np.logical_and(points[:, 1] < filter_up, points[:, 1] > filter_down)
    points = points[condition1]
    condition2 = np.logical_and(points[:, 2] <= 3, points[:, 2] >=1)
    points = points[condition2]

    # index_list, class_k = threshold_cluster(points[:,1], 0.02)
    # for point in points:
    #     if point[2]<4.0:

    # borderPoints=find_border(points)
    # borderPoints = o3d.geometry.PointCloud()
    # borderPoints.paint_uniform_color([0, 1, 0])

    pointClusters = dbscan(points)

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
    # 添加点云
    vis.add_geometry(pcd)

    # for border in points:
    #     borderPoints.points = o3d.utility.Vector3dVector(border)
    #     vis.add_geometry(borderPoints)

    #### ground truth
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
        # 将矩形框加入到窗口中
        vis.add_geometry(line_set)

    #  prediction
    dimension = [5, 2, 1.8]
    input = []  # the data to be written to json file
    objects = []
    temp_objects = []

    # filter y_wide >2
    for cluster in pointClusters:
        temp_object = []
        cluster = np.array(cluster)
        x = np.array(cluster[:, 0])
        y = np.array(cluster[:, 1])
        z = []
        # if z.max()-z.min()<=1:
        #     continue
        for point in origin_points:
            if x.min() <= point[0] <= x.max() and y.min() <= point[1] <= y.max():
                z.append(point[2])
        z = np.array(z)
        centerPoint = [(x.min() + x.max()) / 2, (y.min() + y.max()) / 2, (z.min() + z.max()) / 2]
        bbox = compute_3D_box_cam2(dimension, centerPoint, 1.5)
        bbox_x = np.array(bbox[:, 0])
        bbox_y = np.array(bbox[:, 1])
        bbox_z = np.array(bbox[:, 2])
        translation_x = x.min() - bbox_x.min()
        translation_y = y.min() - bbox_y.min()
        translation_z = z.min() - bbox_z.min()
        if y.min() >= 0 and y.max() >= 0:
            translation_y = y.min() - bbox_y.min()
        elif y.min() < 0 and y.max() < 0:
            translation_y = y.max() - bbox_y.max()
        centerPoint[0] += translation_x
        centerPoint[1] += translation_y
        centerPoint[2] += translation_z
        if centerPoint[1]<=(filter_up-2) and centerPoint[1]>=(filter_down+2):
            temp_object.append(centerPoint)
            boundary = [x.min(), y.min(), z.min()]
            temp_object.append(boundary)
            temp_object.append(len(cluster))
            temp_objects.append(temp_object)
    temp_objects = carBox.merge_tooNear(temp_objects)
    temp_objects = carBox.filter_tooNear(temp_objects)

    ID = 1
    for centerPoint in temp_objects:
        centerPoint = centerPoint[0]
        if (centerPoint[1] + 1) < filter_up and (centerPoint[1] - 1) > filter_down:
            obj = {"shapeType": "cube", "static": False, "box2d": {}, "box3d": {
                "generateMode": 1,
                "center": {
                    "x": 124.87528916911066,
                    "y": 14.046987558348778,
                    "z": 3.178713188295765
                },
                "rotation": {
                    "x": 0,
                    "y": 0,
                    "z": 3.141592653589793
                },
                "isExist": True,
                "isMove": True,
                "content": {
                    "Motion": [
                        "static"
                    ],
                    "ID-2": "",
                    "occulsion": "0",
                    "subclass": "normal",
                    "Lane": "on_Lane/001/L1",
                    "CIPO": "no",
                    "truncation": "0"
                },
                "dimensions": {
                    "length": 5,
                    "width": 1.8,
                    "height": 1.5
                },
                "quality": {
                    "errorType": {
                        "attributeError": [],
                        "targetError": [],
                        "otherError": ""
                    },
                    "changes": {
                        "remark": "",
                        "attribute": [],
                        "target": []
                    },
                    "qualityStatus": "unqualified"
                }
            }, "label": "car", "objectId": 28}

            # Modify the values for "center," "dimensions," and "rotation" for each object
            obj['box3d']['center']['x'] = centerPoint[0] - 2.5  # Modify x-coordinate of center
            obj['box3d']['center']['y'] = centerPoint[1]  # Modify y-coordinate of center
            obj['box3d']['center']['z'] = centerPoint[2]  # Modify z-coordinate of center

            obj['box3d']['dimensions']['length'] = 5  # Modify length
            obj['box3d']['dimensions']['width'] = 2  # Modify width
            obj['box3d']['dimensions']['height'] = 1.8  # Modify height

            obj['box3d']['rotation']['x'] = 0  # Modify x-rotation
            obj['box3d']['rotation']['y'] = 0  # Modify y-rotation
            obj['box3d']['rotation']['z'] = 0  # Modify z-rotation

            obj['objectId'] = ID
            ID += 1
            objects.append(obj)
            # Write the modified JSON back to a file
            # with open('output.json', 'w') as file:
            #     json.dump(data, file, indent=2)

            lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                  [0, 4], [1, 5], [2, 6], [3, 7]])
            bbox = compute_3D_box_cam2(dimension, centerPoint, -1.5)
            colors = np.array([[0, 0, 1] for j in range(len(lines_box))])

            # 创建Bbox候选框对象
            line_set = o3d.geometry.LineSet()
            # 将八个顶点连接次序的信息转换成o3d可以使用的数据类型
            line_set.lines = o3d.utility.Vector2iVector(lines_box)
            # 设置每条线段的颜色
            line_set.colors = o3d.utility.Vector3dVector(colors)
            # 把八个顶点的空间信息转换成o3d可以使用的数据类型
            line_set.points = o3d.utility.Vector3dVector(bbox)
            # 将矩形框加入到窗口中
            vis.add_geometry(line_set)

    # write road border to json
    for border_point in border_points:
        obj = {
            "shapeType": "line",
            "static": False,
            "box2d": {},
            "box3d": {
                "generateMode": 1,
                "coordinates": [
                    [
                        [
                            1.8565764156245046,
                            2.3099910998639093,
                            0
                        ],
                        [
                            55.43836510450844,
                            2.9140614945973184,
                            0
                        ]
                    ]
                ],
                "isExist": True,
                "content": {},
                "quality": {
                    "errorType": {
                        "attributeError": [],
                        "targetError": [],
                        "otherError": ""
                    },
                    "changes": {
                        "remark": "",
                        "attribute": [],
                        "target": []
                    },
                    "qualityStatus": "unqualified"
                }
            },
            "label": "Road-boundary",
            "objectId": 22
        }

        # Modify the values for "center," "dimensions," and "rotation" for each object
        obj['box3d']["coordinates"] = [border_point]  # Modify x-coordinate of center

        obj['objectId'] = ID
        ID += 1
        objects.append(obj)

    with open(jsonPath, 'r', encoding='gbk', errors='replace') as file:
        data = json.load(file)
    data['objects'] = objects

    folder_path = os.path.join("kitty_open3d/output", folder_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open('kitty_open3d/output/{}'.format(jsonPath), 'w') as file:
        json.dump(data, file, indent=2)
    copy_file(pcdPath, os.path.join('kitty_open3d/output', pcdPath))
    vis.run()


if __name__ == "__main__":
    folder_path = "scene_001/002"
    pcd_path = "scene_001/002/1688038376.689710.pcd"
    json_path = "scene_001/002/1688038376.689710.json"
    singleFrame(pcd_path, json_path, folder_path)
