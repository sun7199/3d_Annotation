import json
import os

import numpy as np
import open3d as o3d
import ICP


def read_json(filepath):
    with open(filepath, "rb") as file:
        json_data = json.load(file)
    return json_data


def get_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points represented as dictionaries.

    Parameters:
    - dict1: Dictionary representing the coordinates of the first point.
    - dict2: Dictionary representing the coordinates of the second point.

    Returns:
    - distance: Euclidean distance between the two points.
    """
    # Convert dictionary values to NumPy arrays
    # point1 = np.array(list(dict1.values()))
    # point2 = np.array(list(dict2.values()))

    # Ensure both points have the same dimensionality
    assert point1.shape == point2.shape, "Points must have the same dimensionality."

    # Calculate Euclidean distance
    # distance = np.linalg.norm(point2 - point1)
    distance = abs(point1[1] - point2[1])

    return distance


def draw_bbox(bbox, vis):
    lines_box = np.array([[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7]])
    colors = np.array([[1, 0, 0] for j in range(len(lines_box))])
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


def find_carByICP(sourcePath, targetPath, carIDList):
    transformation_matrix = ICP.icp_registration(sourcePath.replace("json", "pcd"),
                                                 targetPath.replace("json", "pcd"))
    unique_list = [-1 for i in range(0,len(carIDList))]
    distance_list = []

    jsonfile_source = read_json(sourcePath)
    source_objects = jsonfile_source["objects"]

    for carID in carIDList:
        center = None
        for obj in source_objects:
            if obj["objectId"] == carID:
                center = obj['box3d']['center']
                break  # Exit the loop once the carID is found

        if center is None:
            print(f"Warning: CarID {carID} not found in the source file.")
            continue

        jsonfile_target = read_json(targetPath)
        target_objects = jsonfile_target["objects"]
        distances = []
        source_car = np.array(list(center.values()))
        transformed_car = np.dot(transformation_matrix[:3, :3], source_car) + transformation_matrix[:3, 3]

        for obj in target_objects:
            if obj["shapeType"] == "cube":
                distance = get_distance(np.array(list(obj['box3d']['center'].values())), transformed_car)
                distances.append(distance)

        distance_list.append(distances)
    # Find the minimum element and its index
    distance_list=np.array(distance_list)
    for i in range(0,len(carIDList)):
        # Find the minimum element and its indices
        min_value = np.min(distance_list)
        min_indices = np.argwhere(distance_list == min_value)[0]
        print(min_indices)
        unique_list[min_indices[0]]=min_indices[1]
        # Change the corresponding row and column to 100
        distance_list[min_indices[0], :] = 100  # Change row to 100
        distance_list[:, min_indices[1]] = 100  # Change column to 100
    print(unique_list)

    # Visualization code outside the loop
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="point_cloud")
    opt = vis.get_render_option()
    opt.background_color = np.array([255, 255, 255])
    opt.point_size = 3.0
    for i in range(0, len(unique_list)):
        for obj in target_objects:
            if obj["objectId"] == carIDList[i]:
                obj["objectId"] = 0

        target_objects[unique_list[i]]['objectId'] = carIDList[i]

    pcdPath = targetPath.replace("json", "pcd")
    pcd = o3d.io.read_point_cloud(pcdPath)
    vis.add_geometry(pcd)

    for id in carIDList:
        for obj in target_objects:
            if obj["shapeType"]=="cube":
                if obj['objectId']==id:
                    center = obj["box3d"]["center"]
                    dimensions = obj["box3d"]["dimensions"]
                    rotation = obj["box3d"]["rotation"]
                    box_corners = get_3d_box(center, dimensions, rotation)
                    draw_bbox(box_corners, vis)

    vis.run()

    jsonfile_target["objects"] = target_objects

    with open(targetPath, 'w') as file:
        json.dump(jsonfile_target, file, indent=2)


# sourcePath = 'pointcloud_open3d/output/003/003/1688782472.389342.json'
# targetPath = 'pointcloud_open3d/output/003/003/1688782472.890264.json'

def singleFrame(sourcePath, targetPath):
    # read previous frame
    jsonfile = read_json(sourcePath)
    objects = jsonfile["objects"]
    IDlist=[]
    for obj in objects:
        if obj["shapeType"]=="cube":
            IDlist.append(obj["objectId"])
    find_carByICP(sourcePath, targetPath, IDlist)

    # get HOG feature of previous frame

    # box_dimensions = (50, 20, 18)
    # unit = 0.1
    # feature_5 = HOG.getFeature(center_5, box_dimensions, unit)
    # feature_8 = HOG.getFeature(center_8, box_dimensions, unit)
    # feature_10 = HOG.getFeature(center_10, box_dimensions, unit)
    # feature_11 = HOG.getFeature(center_11, box_dimensions, unit)
    # feature_12 = HOG.getFeature(center_12, box_dimensions, unit)
    # result=HOG.compareFeature(feature_12,feature_5)
    # print(result)

    # find near

    # jsonfile = read_json(targetPath)
    # objects = jsonfile["objects"]
    # distances_to5 = []
    # distances_to8 = []
    # distances_to10 = []
    # distances_to11 = []
    # distances_to12 = []
    # for obj in objects:
    #     distance = get_distance(obj['box3d']['center'], center_5)
    #     distances_to5.append(distance)
    # index_5 = np.argmin(distances_to5)
    # # write ID
    # for obj in objects:
    #     if obj["objectId"] == 5:
    #         obj["objectId"] = 0
    # objects[index_5]['objectId'] = 5
    #
    # for obj in objects:
    #     distance = get_distance(obj['box3d']['center'], center_8)
    #     distances_to8.append(distance)
    # index_8 = np.argmin(distances_to8)
    # # write ID
    # for obj in objects:
    #     if obj["objectId"] == 8:
    #         obj["objectId"] = 0
    # objects[index_8]['objectId'] = 8
    #
    # for obj in objects:
    #     distance = get_distance(obj['box3d']['center'], center_10)
    #     distances_to10.append(distance)
    # index_10 = np.argmin(distances_to10)
    # # write ID
    # for obj in objects:
    #     if obj["objectId"] == 10:
    #         obj["objectId"] = 0
    # objects[index_10]['objectId'] = 10
    #
    # for obj in objects:
    #     distance = get_distance(obj['box3d']['center'], center_11)
    #     distances_to11.append(distance)
    # index_11 = np.argmin(distances_to11)
    # # write ID
    # for obj in objects:
    #     if obj["objectId"] == 11:
    #         obj["objectId"] = 0
    # objects[index_11]['objectId'] = 11
    #
    # for obj in objects:
    #     distance = get_distance(obj['box3d']['center'], center_12)
    #     distances_to12.append(distance)
    # index_12 = np.argmin(distances_to12)
    # # write ID
    # for obj in objects:
    #     if obj["objectId"] == 12:
    #         obj["objectId"] = 0
    # objects[index_12]['objectId'] = 12

    # HOG feature compare
    # jsonfile = read_json(targetPath)
    # objects = jsonfile["objects"]
    # similarity = []
    # # find car 5
    # for obj in objects:
    #     # calculate each car feature
    #     center = obj['box3d']['center']
    #     feature = HOG.getFeature(center, (50, 20, 18), 0.1)
    #     result=HOG.compareFeature(feature_5,feature)
    #     similarity.append(result)
    #     print(similarity)
    #     index_5 = np.argmax(similarity)
    # # write ID
    # for obj in objects:
    #     if obj["objectId"] == 5:
    #         obj["objectId"] = 0
    # objects[index_5]['objectId'] = 5
    # find car 8
    # for obj in objects:
    #     similarity = []
    #     # calculate each car feature
    #     center = obj['box3d']['center']
    #     feature = HOG.getFeature(center, (50, 20, 18), 0.1)
    #     result=HOG.compareFeature(feature_8,feature)
    #     similarity.append(result)
    #     index_8 = np.argmax(similarity)
    # # write ID
    # for obj in objects:
    #     if obj["objectId"] == 8:
    #         obj["objectId"] = 0
    # objects[index_8]['objectId'] = 8
    # # find car 10
    # for obj in objects:
    #     similarity = []
    #     # calculate each car feature
    #     center = obj['box3d']['center']
    #     feature = HOG.getFeature(center, (50, 20, 18), 0.1)
    #     result=HOG.compareFeature(feature_10,feature)
    #     similarity.append(result)
    #     index_10 = np.argmax(similarity)
    # # write ID
    # for obj in objects:
    #     if obj["objectId"] == 10:
    #         obj["objectId"] = 0
    # objects[index_10]['objectId'] = 10
    # # find car 11
    # for obj in objects:
    #     similarity = []
    #     # calculate each car feature
    #     center = obj['box3d']['center']
    #     feature = HOG.getFeature(center, (50, 20, 18), 0.1)
    #     result=HOG.compareFeature(feature_11,feature)
    #     similarity.append(result)
    #     index_11 = np.argmax(similarity)
    # # write ID
    # for obj in objects:
    #     if obj["objectId"] == 11:
    #         obj["objectId"] = 0
    # objects[index_11]['objectId'] = 11
    # # find car 12
    # for obj in objects:
    #     similarity = []
    #     # calculate each car feature
    #     center = obj['box3d']['center']
    #     feature = HOG.getFeature(center, (50, 20, 18), 0.1)
    #     result=HOG.compareFeature(feature_12,feature)
    #     similarity.append(result)
    #     index_12 = np.argmax(similarity)
    # # write ID
    # for obj in objects:
    #     if obj["objectId"] == 12:
    #         obj["objectId"] = 0
    # objects[index_12]['objectId'] = 12


if __name__ == "__main__":
    folder_path = "kitty_open3d/output/scene_001/002"
    frame = 2
    for root, dirs, files in os.walk(folder_path):
        if files:
            if files[0].endswith(".json"):
                for i in range(0, len(files), 2):
                    file1 = files[i]
                    file2 = files[i + 2] if i + 1 < len(files) else None

                    sourcePath = os.path.join(root, file1)
                    sourcePath = sourcePath.replace("\\", "/")
                    targetPath = os.path.join(root, file2) if file2 else None
                    targetPath = targetPath.replace("\\", "/")
                    print("Frame:", frame)
                    frame += 1
                    singleFrame(sourcePath, targetPath)

# draw bbox first frame
# pcdPath = '003/003/1688782470.889551.pcd'
# pcd = o3d.io.read_point_cloud(pcdPath)
# vis = o3d.visualization.Visualizer()
# # 创建窗口，设置窗口标题
# vis.create_window(window_name="point_cloud")
# # 设置点云渲染参数
# opt = vis.get_render_option()
# # 设置背景色（这里为白色）
# opt.background_color = np.array([255, 255, 255])
# # 设置渲染点的大小
# opt.point_size = 3.0
# vis.add_geometry(pcd)
# center = initial_car5["center"]
# dimensions = initial_car5["dimensions"]
# rotation = initial_car5["rotation"]
# box_corners = get_3d_box(center, dimensions, rotation)
# draw_bbox(box_corners)
# center = initial_car8["center"]
# dimensions = initial_car8["dimensions"]
# rotation = initial_car8["rotation"]
# box_corners = get_3d_box(center, dimensions, rotation)
# draw_bbox(box_corners)
# center = initial_car11["center"]
# dimensions = initial_car11["dimensions"]
# rotation = initial_car11["rotation"]
# box_corners = get_3d_box(center, dimensions, rotation)
# draw_bbox(box_corners)
# center = initial_car12["center"]
# dimensions = initial_car12["dimensions"]
# rotation = initial_car12["rotation"]
# box_corners = get_3d_box(center, dimensions, rotation)
# draw_bbox(box_corners)
# vis.run()
