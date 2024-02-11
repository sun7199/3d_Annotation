import open3d as o3d
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

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

def draw_lines_on_point_cloud(point_cloud, left_edge_y, right_edge_y, avg_z):
    # 创建线段
    lines = []
    min_x = np.min(np.asarray(point_cloud.points)[:, 0])
    max_x = np.max(np.asarray(point_cloud.points)[:, 0])

    # 在最左边和最右边的道路边沿位置绘制线
    lines.append([min_x, left_edge_y, avg_z])
    lines.append([max_x, left_edge_y, avg_z])
    lines.append([min_x, right_edge_y, avg_z])
    lines.append([max_x, right_edge_y, avg_z])
    print('left_edge_y',left_edge_y)
    print('right_edge_y',right_edge_y)
    # 创建线集对象
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(lines),
        lines=o3d.utility.Vector2iVector([[0, 1], [2, 3]])
    )
    line_set.paint_uniform_color([1, 0, 0])  # 红色线条

    # 可视化原始点云和线条
    o3d.visualization.draw_geometries([point_cloud, line_set])

def getEdge(pcdPath):
    # 主程序
    # 加载点云
    pcd = load_point_cloud(pcdPath)

    # 道路边沿检测
    x_threshold = 10.0
    y_max_width = 0.2
    z_min = -1.0
    z_max = 1.0
    min_points = 2
    edge_points = filter_road_edge_points(pcd, x_threshold, y_max_width, z_min, z_max, min_points)
    # # 标记边沿点云在原始点云中
    # mark_road_edges_in_point_cloud(pcd, edge_points, color=[1, 0, 0])  # 使用红色标记边沿点云
    # o3d.visualization.draw_geometries([pcd])
    # 初步边沿位置
    left_edge_y, right_edge_y = find_edge_lines_with_polynomial_regression(edge_points)

    # 寻找附近点云
    # 定义搜索范围
    min_x=0
    max_x=10
    min_z=-1.0
    max_z=1.5
    x_range = (min_x, max_x)
    y_range = 1.5  # 例如，0.5米
    z_range = (min_z, max_z)
    left_nearby_points = find_nearby_points(edge_points, left_edge_y, x_range, y_range, z_range)
    right_nearby_points = find_nearby_points(edge_points, right_edge_y, x_range, y_range, z_range)

    # 重新计算边沿
    refined_left_edge_y = refine_edge_line_with_regression(left_nearby_points)
    refined_right_edge_y = refine_edge_line_with_regression(right_nearby_points)

    # 计算平均Z坐标
    avg_z = np.mean(edge_points[:, 2])

    return refined_left_edge_y,refined_right_edge_y

    # 绘制道路边沿
    draw_lines_on_point_cloud(pcd, refined_left_edge_y, refined_right_edge_y, avg_z)
    # draw_pca_lines_on_point_cloud(pcd, edge_points)

