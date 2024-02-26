import AlphaShape
import numpy as np
import open3d as o3d
from sklearn.linear_model import LinearRegression


def get_roadBoarder(points):
    # up_boarder, down_boarder = AlphaShape.find_roadBorder(points)
    temp = []
    border_points = []

    condition = np.logical_and(points[:, 2] < 0.3, points[:, 2] > 0)
    points = points[condition]

    filter_up = np.max(points[:, 1])
    filter_down = np.min(points[:, 1])
    filter_right = np.max(points[:, 0])

    up_line = [[0, filter_up, 0], [100, filter_up, 0]]
    down_line = [[0, filter_down, 0], [100, filter_down, 0]]

    border_points.append(up_line)
    border_points.append(down_line)
    return border_points, filter_up, filter_down, filter_right


def draw_boarder(points):
    # Flatten the 2D array into a list of points
    flat_points = points.reshape(-1, 2)

    # Fit a linear regression line to the points
    model = LinearRegression().fit(flat_points[:, 0].reshape(-1, 1), flat_points[:, 1])

    # Calculate the endpoints of the line based on the data range
    x_min, x_max = flat_points[:, 0].min(), flat_points[:, 0].max()
    y_min, y_max = model.predict([[x_min]]), model.predict([[x_max]])

    start_point = [x_min, y_min, 0]
    end_point = [x_max, y_max, 0]

    return start_point, end_point


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("scene_001/002/1688038395.489439.pcd")
    points = np.array(pcd.points)
    border_points, filter_up, filter_down,filter_right= get_roadBoarder(points)
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 创建窗口，设置窗口标题
    vis.create_window(window_name="point_cloud")
    # 设置点云渲染参数
    opt = vis.get_render_option()
    # 设置背景色（这里为白色）
    opt.background_color = np.array([255, 255, 255])
    # 设置渲染点的大小
    opt.point_size = 1.0
    # 添加点云
    vis.add_geometry(pcd)

    lines_box = np.array([[0, 1]])
    # 设置点与点之间线段的颜色
    colors = np.array([[1, 0, 0] for j in range(len(lines_box))])
    # 创建Bbox候选框对象
    line_set = o3d.geometry.LineSet()
    # 将八个顶点连接次序的信息转换成o3d可以使用的数据类型
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    # 设置每条线段的颜色
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # 把八个顶点的空间信息转换成o3d可以使用的数据类型
    line_set.points = o3d.utility.Vector3dVector(border_points[0])
    # 将矩形框加入到窗口中
    vis.add_geometry(line_set)

    lines_box = np.array([[0, 1]])
    # 设置点与点之间线段的颜色
    colors = np.array([[1, 0, 0] for j in range(len(lines_box))])
    # 创建Bbox候选框对象
    line_set = o3d.geometry.LineSet()
    # 将八个顶点连接次序的信息转换成o3d可以使用的数据类型
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    # 设置每条线段的颜色
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # 把八个顶点的空间信息转换成o3d可以使用的数据类型
    line_set.points = o3d.utility.Vector3dVector(border_points[1])
    # 将矩形框加入到窗口中
    vis.add_geometry(line_set)

    vis.run()
