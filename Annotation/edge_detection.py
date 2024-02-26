import open3d as o3d
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy import stats
import json
import queue
# 加载点云
def load_point_cloud(filename):
    return o3d.io.read_point_cloud(filename)
def visualize_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])
# 初步道路边沿检测
def filter_road_edge_points(point_cloud, x_threshold=0.5, y_max_width=1.0, z_min=-1.0, z_max=1.0, min_points=100):
    points = np.asarray(point_cloud.points)
    points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
    points = points[points[:, 0].argsort()]

    edge_points = []
    start_index = 0
    for i in range(1, len(points)):
        x_diff = points[i, 0] - points[i - 1, 0]
        y_diff = abs(points[i, 1] - points[i - 1, 1])
        if x_diff > x_threshold or y_diff > y_max_width:
            if i - start_index > min_points:
                segment = points[start_index:i]
                edge_points.append(segment)
            start_index = i
    return np.vstack(edge_points) if edge_points else np.array([]).reshape(0, 3)


def find_edge_lines(point_cloud):
    # 假设 point_cloud 已经是一个 NumPy 数组
    points = point_cloud

    # 分割点云到四个象限
    quadrants = {
        1: points[(points[:, 0] >= 0) & (points[:, 1] >= 0)],
        2: points[(points[:, 0] < 0) & (points[:, 1] >= 0)],
        3: points[(points[:, 0] < 0) & (points[:, 1] < 0)],
        4: points[(points[:, 0] >= 0) & (points[:, 1] < 0)]
    }

    center_points = []

    for quadrant, q_points in quadrants.items():
        if len(q_points) == 0:
            continue

        # 计算每个点到原点的距离
        distances = np.linalg.norm(q_points, axis=1)

        # 确定距离最远的点云集合
        far_points = q_points[distances > np.percentile(distances, 90)]  # 使用90%分位数作为阈值

        if len(far_points) == 0:
            continue

        # 计算远点集合的中心点
        center_point = np.mean(far_points, axis=0)
        center_points.append(center_point)

    # 检查是否有足够的中心点形成边沿线
    if len(center_points) < 4:
        return None, None  # 不足四个象限的中心点

    # 按x坐标排序中心点
    center_points = sorted(center_points, key=lambda x: x[0])

    left_line = np.array([center_points[0], center_points[1]])  # x轴左侧的两个中心点
    right_line = np.array([center_points[2], center_points[3]])  # x轴右侧的两个中心点

    return left_line, right_line

# 多项式回归边沿检测
def find_edge_lines_with_polynomial_regression(edge_points, num_groups=100, degree=2):
    sorted_points = edge_points[np.argsort(edge_points[:, 1])]
    group_size = len(sorted_points) // num_groups
    group_edges = []

    for i in range(num_groups):
        start_index = i * group_size
        end_index = start_index + group_size
        group = sorted_points[start_index:end_index]
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(group[:, 0].reshape(-1, 1))
        model = LinearRegression()
        model.fit(X_poly, group[:, 1])
        X_predict = poly.transform([[np.mean(group[:, 0])]])
        edge_y = model.predict(X_predict)[0]
        group_edges.append(edge_y)

    left_edge_y = min(group_edges)
    right_edge_y = max(group_edges)
    return left_edge_y, right_edge_y

def find_x_extremes_at_y(edge_points, y_value, tolerance=0.5):
    # 筛选出靠近特定Y值的点
    nearby_points = edge_points[np.abs(edge_points[:, 1] - y_value) <= tolerance]
    # 找到这些点中X的最小值和最大值
    x_min = np.min(nearby_points[:, 0])
    x_max = np.max(nearby_points[:, 0])
    return x_min, x_max

# 寻找直线附近的点云
def find_nearby_points(edge_points, line_y, x_range, y_range, z_range):
    # 筛选出靠近直线的点云
    nearby_points = edge_points[
        (edge_points[:, 0] >= x_range[0]) & (edge_points[:, 0] <= x_range[1]) &
        (edge_points[:, 1] >= line_y - y_range) & (edge_points[:, 1] <= line_y + y_range) &
        (edge_points[:, 2] >= z_range[0]) & (edge_points[:, 2] <= z_range[1])
    ]
    return nearby_points


# 重新计算道路边沿
def refine_edge_line_with_regression(nearby_points, degree=2):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(nearby_points[:, 0].reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, nearby_points[:, 1])
    mean_x = np.mean(nearby_points[:, 0])
    X_predict = poly.transform([[mean_x]])
    refined_edge_y = model.predict(X_predict)[0]
    return refined_edge_y

# 绘制道路边沿
def draw_lines_on_point_cloud(point_cloud, left_edge_y, right_edge_y, avg_z):
    lines = []
    min_x = np.min(np.asarray(point_cloud.points)[:, 0])
    max_x = np.max(np.asarray(point_cloud.points)[:, 0])
    lines.append([min_x, left_edge_y, avg_z])
    lines.append([max_x, left_edge_y, avg_z])
    lines.append([min_x, right_edge_y, avg_z])
    lines.append([max_x, right_edge_y, avg_z])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(lines),
        lines=o3d.utility.Vector2iVector([[0, 1], [2, 3]])
    )
    line_set.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([point_cloud, line_set])

def draw_pca_lines_on_point_cloud(point_cloud, edge_points):
    pca = PCA(n_components=2)
    pca.fit(edge_points[:, :2])  # 只考虑X和Y坐标
    main_direction = pca.components_[0]  # 主方向
    main_direction_3d = np.append(main_direction, 0)  # 转换为3D

    lines = []
    for edge_y in [np.min(edge_points[:, 1]), np.max(edge_points[:, 1])]:
        center_point = np.array([np.mean(edge_points[:, 0]), edge_y, np.mean(edge_points[:, 2])])
        line_start = center_point - main_direction_3d * 5  # 线长度可调整
        line_end = center_point + main_direction_3d * 5
        lines.append(line_start)
        lines.append(line_end)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(lines),
        lines=o3d.utility.Vector2iVector([[0, 1], [2, 3]])
    )
    line_set.paint_uniform_color([1, 0, 0])  # 红色线条
    o3d.visualization.draw_geometries([point_cloud, line_set])
def mark_road_edges_in_point_cloud(original_point_cloud, edge_points, color=[1, 0, 0]):
    # 将原始点云转换为NumPy数组
    original_points = np.asarray(original_point_cloud.points)
    if not original_point_cloud.has_colors():
        original_point_cloud.colors = o3d.utility.Vector3dVector(np.zeros_like(original_points))
    original_colors = np.asarray(original_point_cloud.colors)

    # 遍历原始点云，将边沿点云标记为特定颜色
    for point in edge_points:
        distances = np.linalg.norm(original_points - point, axis=1)
        closest_index = np.argmin(distances)
        original_colors[closest_index] = color

    # 更新原始点云的颜色
    original_point_cloud.colors = o3d.utility.Vector3dVector(original_colors)
def find_line_from_points(points):
    model = LinearRegression()
    X = points[:, [0]]  # X坐标
    Y = points[:, [1]]  # Y坐标
    model.fit(X, Y)
    # 直线的斜率和截距
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept

def get_edge_lines(edge_points):
    # 分别找到最左边和最右边的点云
    leftmost_points = edge_points[edge_points[:, 1] == np.min(edge_points[:, 1])]
    rightmost_points = edge_points[edge_points[:, 1] == np.max(edge_points[:, 1])]

    # 找到两条直线
    left_line_slope, left_line_intercept = find_line_from_points(leftmost_points)
    right_line_slope, right_line_intercept = find_line_from_points(rightmost_points)

    return (left_line_slope, left_line_intercept), (right_line_slope, right_line_intercept)
def generate_line_points(slope, intercept, x_min, x_max):
    # 确保斜率和截距是单个数值
    slope = np.squeeze(slope)
    intercept = np.squeeze(intercept)

    # 根据斜率和截距生成直线的两个端点
    y_min = slope * x_min + intercept
    y_max = slope * x_max + intercept
    return np.array([[x_min, y_min, 0], [x_max, y_max, 0]])

def calculate_new_y(x1, y1, x2, y2, x_new):
    """根据线段的两个端点和新的 x 坐标计算新的 y 坐标。"""
    slope = (y2 - y1) / (x2 - x1)  # 计算斜率
    y_new = slope * (x_new - x1) + y1  # 使用线性方程计算新的 y 坐标
    return y_new

# def draw_lines_on_point_cloud(point_cloud, centers_close, centers_l, centers_r, avg_z):
#     # 创建线段
#     lines = []
#     min_x = np.min(np.asarray(point_cloud.points)[:, 0])
#     max_x = np.max(np.asarray(point_cloud.points)[:, 0])
#     lines.append([centers_close[0][0], centers_close[0][1], avg_z])
#     lines.append([centers_close[1][0], centers_close[1][1], avg_z])
#     lines.append([centers_l[0][0], centers_l[0][1], avg_z])
#     lines.append([centers_l[1][0], centers_l[1][1], avg_z])
#     lines.append([centers_r[0][0], centers_r[0][1], avg_z])
#     lines.append([centers_r[1][0], centers_r[1][1], avg_z])
#     # 创建线集对象
#     line_set = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(lines),
#         lines=o3d.utility.Vector2iVector([[0, 1], [2, 3], [4, 5]])
#     )
#     line_set.paint_uniform_color([1, 0, 0])  # 红色线条
#
#     #线段延伸
#     extended_lines = []
#     for i in range(0, len(lines), 2):
#         x1, y1, z = lines[i]
#         x2, y2, _ = lines[i + 1]
#
#         # 计算延伸到 x_min 和 x_max 的新 y 坐标
#         y_new_at_x_min = calculate_new_y(x1, y1, x2, y2, min_x)
#         y_new_at_x_max = calculate_new_y(x1, y1, x2, y2, max_x)
#
#         # 添加延伸后的线段端点
#         extended_lines.append([min_x, y_new_at_x_min, z])
#         extended_lines.append([max_x, y_new_at_x_max, z])
#         print('extended_lines',extended_lines)
#
#     # 使用 extended_lines 创建 LineSet 对象
#     line_set = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(extended_lines),
#         lines=o3d.utility.Vector2iVector([[0, 1], [2, 3], [4, 5]])
#     )
#     #线段延伸
#     line_set.paint_uniform_color([1, 0, 0])  # 红色线条
#     # 可视化原始点云和线条
#     o3d.visualization.draw_geometries([point_cloud, line_set])
def calculate_new_y_center(x1, y1, x2, y2, x_new, slope):
    y_new = slope * (x_new - x1) + y1
    return y_new
def calculate_new_y(x1, y1, x2, y2, x_new):
    slope = (y2 - y1) / (x2 - x1)
    y_new = slope * (x_new - x1) + y1
    return y_new, slope

def sample_points_on_line(line_start, line_end, num_points):
    return np.linspace(line_start, line_end, num_points)

def draw_lines_on_point_cloud(point_cloud, centers_close, centers_l, centers_r, avg_z):
    lines = []
    avg_z=0
    min_x = np.min(np.asarray(point_cloud.points)[:, 0])
    max_x = np.max(np.asarray(point_cloud.points)[:, 0])
    lines.append([centers_l[0][0], centers_l[0][1], avg_z])
    lines.append([centers_l[1][0], centers_l[1][1], avg_z])
    lines.append([centers_r[0][0], centers_r[0][1], avg_z])
    lines.append([centers_r[1][0], centers_r[1][1], avg_z])
    lines.append([centers_close[0][0], centers_close[0][1], avg_z])
    lines.append([centers_close[1][0], centers_close[1][1], avg_z])
    # 保留原始点云的点和颜色
    all_points = np.asarray(point_cloud.points)
    all_colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None
    centers_slope=0
    # 为线条上的点添加红色
    num_points_per_line = 1000  # 每条线的点数
    sampled_lines_points = []  # 用于存储所有线条上的采样点
    print_sampled_line_points = []
    for i in range(0, len(lines), 2):
        x1, y1, z = lines[i]
        x2, y2, _ = lines[i + 1]
        if i == 0:
            y_new_at_x_min, centers_slope = calculate_new_y(x1, y1, x2, y2, min_x)
            y_new_at_x_max, centers_slope = calculate_new_y(x1, y1, x2, y2, max_x)
            extended_line_start = [min_x, y_new_at_x_min, z]
            extended_line_end = [max_x, y_new_at_x_max, z]
            sampled_line_points = sample_points_on_line(extended_line_start, extended_line_end, num_points_per_line)
            print_sampled_line_points.append(sample_points_on_line(extended_line_start, extended_line_end, 4))
            print('print_sampled_line_points', print_sampled_line_points[0])
        elif i == 2:
            y_new_at_x_min, centers_slope = calculate_new_y(x1, y1, x2, y2, min_x)
            y_new_at_x_max, centers_slope = calculate_new_y(x1, y1, x2, y2, max_x)
            extended_line_start = [min_x, y_new_at_x_min, z]
            extended_line_end = [max_x, y_new_at_x_max, z]
            sampled_line_points = sample_points_on_line(extended_line_start, extended_line_end, num_points_per_line)
            print_sampled_line_points.append(sample_points_on_line(extended_line_start, extended_line_end, 3))
            print('print_sampled_line_points', print_sampled_line_points[1])
        elif i ==4:
            y_new_at_x_min = calculate_new_y_center(x1, y1, x2, y2, min_x, centers_slope)
            # print('centers_slope',centers_slope)
            y_new_at_x_max = calculate_new_y_center(x1, y1, x2, y2, max_x, centers_slope)
            extended_line_start = [min_x, y_new_at_x_min, z]
            extended_line_end = [max_x, y_new_at_x_max, z]
            sampled_line_points = sample_points_on_line(extended_line_start, extended_line_end, num_points_per_line)
            print_sampled_line_points.append(sample_points_on_line(extended_line_start, extended_line_end, 8))
            print('print_sampled_line_points', print_sampled_line_points[2])
        line_colors = np.tile([1, 0, 0], (sampled_line_points.shape[0], 1))  # 红色

        # 将线条点和颜色合并到总点云中
        all_points = np.vstack((all_points, sampled_line_points))
        sampled_lines_points.extend(sampled_line_points)  # 添加到采样点列表
        if all_colors is not None:
            all_colors = np.vstack((all_colors, line_colors))
        else:
            all_colors = line_colors

    # 创建带颜色的点云
    colored_point_cloud = o3d.geometry.PointCloud()
    colored_point_cloud.points = o3d.utility.Vector3dVector(all_points)
    if all_colors is not None:
        colored_point_cloud.colors = o3d.utility.Vector3dVector(all_colors)

    # 可视化和保存点云
    o3d.visualization.draw_geometries([colored_point_cloud])
    o3d.io.write_point_cloud("extended_colored_point_cloud.pcd", colored_point_cloud)
# def calculate_new_y(x1, y1, x2, y2, x_new):
#     slope = (y2 - y1) / (x2 - x1)
#     y_new = slope * (x_new - x1) + y1
#     return y_new
#
# def sample_points_on_line(line_start, line_end, num_points):
#     return np.linspace(line_start, line_end, num_points)
#
# def draw_lines_on_point_cloud(point_cloud, centers_close, centers_l, centers_r, avg_z):
#     lines = []
#     min_x = np.min(np.asarray(point_cloud.points)[:, 0])
#     max_x = np.max(np.asarray(point_cloud.points)[:, 0])
#     lines.append([centers_close[0][0], centers_close[0][1], avg_z])
#     lines.append([centers_close[1][0], centers_close[1][1], avg_z])
#     lines.append([centers_l[0][0], centers_l[0][1], avg_z])
#     lines.append([centers_l[1][0], centers_l[1][1], avg_z])
#     lines.append([centers_r[0][0], centers_r[0][1], avg_z])
#     lines.append([centers_r[1][0], centers_r[1][1], avg_z])
#
#     # 保留原始点云的点和颜色
#     all_points = np.asarray(point_cloud.points)
#     all_colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None
#
#     # 为线条上的点添加红色
#     num_points_per_line = 1000  # 每条线的点数
#     sampled_lines_points = []  # 用于存储所有线条上的采样点
#     for i in range(0, len(lines), 2):
#         x1, y1, z = lines[i]
#         x2, y2, _ = lines[i + 1]
#         y_new_at_x_min = calculate_new_y(x1, y1, x2, y2, min_x)
#         y_new_at_x_max = calculate_new_y(x1, y1, x2, y2, max_x)
#         extended_line_start = [min_x, y_new_at_x_min, z]
#         extended_line_end = [max_x, y_new_at_x_max, z]
#         sampled_line_points = sample_points_on_line(extended_line_start, extended_line_end, num_points_per_line)
#         line_colors = np.tile([1, 0, 0], (sampled_line_points.shape[0], 1))  # 红色
#
#         # 将线条点和颜色合并到总点云中
#         all_points = np.vstack((all_points, sampled_line_points))
#         sampled_lines_points.extend(sampled_line_points)  # 添加到采样点列表
#         if all_colors is not None:
#             all_colors = np.vstack((all_colors, line_colors))
#         else:
#             all_colors = line_colors
#
#     # 打印采样线条上的点
#     for i, line_points in enumerate(sampled_lines_points, 1):
#         print(f"Line {i}: Points")
#         for point in line_points:
#             print(point)
#
#     # 创建带颜色的点云
#     colored_point_cloud = o3d.geometry.PointCloud()
#     colored_point_cloud.points = o3d.utility.Vector3dVector(all_points)
#     if all_colors is not None:
#         colored_point_cloud.colors = o3d.utility.Vector3dVector(all_colors)
#
#     # 可视化和保存点云
#     o3d.visualization.draw_geometries([colored_point_cloud])
#     o3d.io.write_point_cloud("extended_colored_point_cloud.pcd", colored_point_cloud)

def draw_edge_lines_with_point_cloud(point_cloud_o3d, left_line, right_line):
    # 创建 LineSet 对象
    lines = o3d.geometry.LineSet()

    # 将线的端点添加到 LineSet
    points = []
    lines_indices = []
    if left_line is not None:
        idx_start = len(points)
        points.extend(left_line)
        lines_indices.append([idx_start, idx_start + 1])

    if right_line is not None:
        idx_start = len(points)
        points.extend(right_line)
        lines_indices.append([idx_start, idx_start + 1])

    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(lines_indices)

    # 设置线的颜色 (例如：红色和绿色)
    colors = [[1, 0, 0] for i in range(len(lines.lines))]
    lines.colors = o3d.utility.Vector3dVector(colors)

    # 绘制点云和线
    o3d.visualization.draw_geometries([point_cloud_o3d, lines])

# def cluster_and_find_centers(point_cloud, y_threshold, n_clusters=4):
#     # 去除靠近x轴的点
#     filtered_points = point_cloud[abs(point_cloud[:, 1]) > y_threshold]
#
#     # 应用K-means聚类
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(filtered_points)
#     labels = kmeans.labels_
#
#     # 找到每个簇的中心点
#     centers = kmeans.cluster_centers_
#
#     for i, center in enumerate(centers):
#         print(f"Cluster {i} center: {center}")
#
#     return centers, labels

def remove_noise(points, nb_neighbors, std_ratio):
    # 将 numpy 数组转换为 open3d 的 PointCloud 对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 使用统计分析进行噪声移除
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)

    # 选择保留的点
    inlier_cloud = pcd.select_by_index(ind)
    return np.asarray(inlier_cloud.points)


def linear_fit_and_extreme_points(points):
    # 将点云数据分解为x和y
    x = points[:, 0]
    y = points[:, 1]

    # 执行线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # 选择两个极端点
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = slope * x_min + intercept, slope * x_max + intercept

    return (x_min, y_min), (x_max, y_max)
def polynomial_fit_and_extreme_points(points, degree):
    # 将点云数据分解为x和y
    x = points[:, 0]
    y = points[:, 1]

    # 进行多项式拟合
    coefficients = np.polyfit(x, y, degree)
    poly = np.poly1d(coefficients)

    # 选择两个极端点
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = poly(x_min), poly(x_max)

    return (x_min, y_min), (x_max, y_max)
def preprocess_and_cluster(point_cloud, y_threshold, y_remove_threshold, left_edge_y, n_clusters=4):
    # 选择靠近x轴的点进行聚类
    print('left_edge_y',left_edge_y)
    if left_edge_y < 0:
        close_to_axis_points = point_cloud[(y_remove_threshold+6 >= point_cloud[:, 1]) & (point_cloud[:, 1] >= y_remove_threshold)]
    else:
        close_to_axis_points = point_cloud[(y_remove_threshold+6 >= point_cloud[:, 1]) & (point_cloud[:, 1] >= y_remove_threshold)]
    # close_to_axis_points = point_cloud[abs(point_cloud[:, 1]) <= y_remove_threshold]
    # nb_neighbors = 20  # 考虑的邻居数量
    # std_ratio = 2.0  # 标准差倍数
    # close_to_axis_points = remove_noise(close_to_axis_points, nb_neighbors, std_ratio)
    # degree = 3  # 多项式次数，可以根据需求调整
    # centers_close = polynomial_fit_and_extreme_points(close_to_axis_points,degree)
    # 对靠近x轴的点应用K-means聚类
    kmeans_close = KMeans(n_clusters=2, random_state=1)
    kmeans_close.fit(close_to_axis_points)
    centers_close = kmeans_close.cluster_centers_
    # visualize_point_cloud(close_to_axis_points)
    # 打印靠近x轴的两个中心点
    for i, center in enumerate(centers_close):
        print(f"Close to x-axis cluster {i} center: {center}")

    # 去除靠近x轴的点
    if left_edge_y < 0:
        filtered_points_l = point_cloud[y_threshold+6 <= point_cloud[:, 1]]
        filtered_points_r = point_cloud[(point_cloud[:, 1] >= y_threshold-12) & (point_cloud[:, 1] <= y_threshold-9)]
    else:
        filtered_points_l = point_cloud[-y_threshold+6 <= point_cloud[:, 1]]
        filtered_points_r = point_cloud[point_cloud[:, 1]<=y_threshold]
    # filtered_points = point_cloud[abs(point_cloud[:, 1]) > y_threshold]
    # visualize_point_cloud(filtered_points_r)
    # centers_l=fit_line_and_find_centers(filtered_points_l)
    # centers_r=fit_line_and_find_centers(filtered_points_r)
    nb_neighbors = 20  # 考虑的邻居数量
    std_ratio = 2.0  # 标准差倍数
    filtered_points_l = remove_noise(filtered_points_l, nb_neighbors, std_ratio)
    # visualize_point_cloud(filtered_points_l)
    degree = 3  # 多项式次数，可以根据需求调整
    centers_l = polynomial_fit_and_extreme_points(filtered_points_l,degree)
    nb_neighbors = 20  # 考虑的邻居数量
    std_ratio = 2.0  # 标准差倍数
    filtered_points_r = remove_noise(filtered_points_r, nb_neighbors, std_ratio)
    # visualize_point_cloud(filtered_points_l)
    kmeans_close = KMeans(n_clusters=2, random_state=1)
    kmeans_close.fit(filtered_points_r)
    # visualize_point_cloud(filtered_points_r)
    centers_r = kmeans_close.cluster_centers_
    print('centers_l',centers_l)
    print('centers_r', centers_r)
    # # 应用K-means聚类
    # kmeans = KMeans(n_clusters=n_clusters)
    # kmeans.fit(filtered_points)
    # labels = kmeans.labels_
    #
    # # 找到每个簇的中心点
    # centers = kmeans.cluster_centers_
    #
    # for i, center in enumerate(centers):
    #     print(f"Cluster {i} center: {center}")
    #
    # return centers_close, centers, labels
    return centers_close, centers_l, centers_r

# 主程序
if __name__=="__main__":
    # 加载点云
    # pcd = load_point_cloud('data\\report数据集\\102_lidar_33-1\ext\ext_pcd_ego\车多\scene_006\\003-transfered\\1688782475.889312.pcd')
    pcd = o3d.io.read_point_cloud("scene_001/002/1688038379.689443.pcd")
    # 道路边沿检测
    x_threshold = 1.0#10.0
    y_max_width = 0.2
    z_min = -1.0
    z_max = 1.0
    min_points = 2#2
    edge_points = filter_road_edge_points(pcd, x_threshold, y_max_width, z_min, z_max, min_points)
    # # 标记边沿点云在原始点云中
    # mark_road_edges_in_point_cloud(pcd, edge_points, color=[1, 0, 0])  # 使用红色标记边沿点云
    # o3d.visualization.draw_geometries([pcd])
    # 初步边沿位置
    # left_edge_y, right_edge_y = find_edge_lines_with_polynomial_regression(edge_points)

    # 使用多项式回归找到左侧和右侧的Y值
    left_edge_y, right_edge_y = find_edge_lines_with_polynomial_regression(edge_points)
    # print('left_edge_y',left_edge_y)
    # print('right_edge_y',right_edge_y)
    # 找到左侧和右侧Y值处X的极端值
    # x_extremes_left = find_x_extremes_at_y(edge_points, left_edge_y)
    # x_extremes_right = find_x_extremes_at_y(edge_points, right_edge_y)
    # # print('x_extremes_left',x_extremes_left)
    # # print('x_extremes_right',x_extremes_right)
    # # 寻找附近点云
    # # 定义搜索范围
    # min_x=0
    # max_x=10
    # min_z=-1.0
    # max_z=1.5
    # x_range = (min_x, max_x)
    # y_range = 1.5  # 例如，0.5米
    # z_range = (min_z, max_z)
    # left_nearby_points = find_nearby_points(edge_points, left_edge_y, x_range, y_range, z_range)
    # right_nearby_points = find_nearby_points(edge_points, right_edge_y, x_range, y_range, z_range)
    #
    # # 重新计算边沿
    # refined_left_edge_y = refine_edge_line_with_regression(left_nearby_points)
    # refined_right_edge_y = refine_edge_line_with_regression(right_nearby_points)

    # 计算平均Z坐标
    avg_z = np.mean(edge_points[:, 2])

    # print('refined_left_edge_y',refined_left_edge_y)
    # print('refined_right_edge_y',refined_right_edge_y)
    # 绘制道路边沿
    # centers_close, centers, labels = preprocess_and_cluster(edge_points,-2, 2,left_edge_y)
    centers_close, centers_l, centers_r = preprocess_and_cluster(edge_points,3, 2,left_edge_y)
    draw_lines_on_point_cloud(pcd, centers_close, centers_l, centers_r, avg_z)
    # draw_lines_on_point_cloud(pcd, refined_left_edge_y, refined_right_edge_y, avg_z)
    # draw_pca_lines_on_point_cloud(pcd, edge_points)