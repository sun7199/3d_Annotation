import CSF
import open3d as o3d
import os
import numpy as np
from scipy.spatial import Delaunay

pcdfile = "003/003/1688782470.889551.pcd"
pcd = o3d.io.read_point_cloud(pcdfile)
points=np.array(pcd.points)
file = open("samp52.bin", "w+")
np.savetxt('samp52.txt', points)
csf = CSF.CSF()
csf.readPointsFromFile('samp52.txt')

csf.params.bSloopSmooth = False
csf.params.cloth_resolution = 0.5

ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
non_ground = CSF.VecInt()  # a list to indicate the index of non-ground points after calculation
csf.do_filtering(ground, non_ground)  # do actual filtering.
csf.savePoints(ground, "ground.txt")
ground_points = np.loadtxt("ground.txt")
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
ground_pcd = o3d.geometry.PointCloud()
condition = np.logical_and(ground_points[:, 2] < 0.3,ground_points[:, 2] > 0)
ground_points = ground_points[condition]
ground_pcd.points = o3d.utility.Vector3dVector(ground_points)

# Compute Delaunay triangulation
tri = Delaunay(ground_points)

# Calculate the circumradius of each simplex
circum_radii = np.sqrt(np.sum(tri.transform[:, :-1, -1] ** 2, axis=1))

# Select the simplices with circumradius smaller than alpha
alpha = 0.1
selected_simplices = tri.simplices[circum_radii < alpha]

# Extract unique edges from selected simplices
edges = set()
for simplex in selected_simplices:
    for i in range(3):
        edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
        edges.add(edge)
print(edges)

# Create Open3D lineset object
lines = []
for edge in edges:
    lines.append([edge[0], edge[1]])
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)
print(lines)

# Visualize the point cloud and alpha shape edges
# vis.add_geometry(ground_pcd)

# 添加点云
vis.add_geometry(line_set)

vis.run()
