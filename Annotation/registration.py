import open3d as o3d
import numpy as np
import os


def perform_icp(source_path, target_path, max_iterations=50, distance_threshold=0.02):
    # Load point clouds
    source_cloud = o3d.io.read_point_cloud(source_path)
    target_cloud = o3d.io.read_point_cloud(target_path)

    # Perform ICP registration
    reg_p2p = o3d.registration.registration_icp(
        source_cloud, target_cloud, distance_threshold, np.identity(4),
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    # Apply the transformation matrix to the source point cloud
    source_cloud.transform(reg_p2p.transformation)

    return source_cloud


if __name__ == "__main__":
    folder_path = "003/003"
    for root, dirs, files in os.walk(folder_path):
        if files:
            if files[0].endswith("json"):
                target_path = files[9]
                target_cloud = o3d.io.read_point_cloud(target_path)
                target_points = np.asarray(target_cloud.points)
                for i in range(0, 20, 2):
                    file1 = files[i]
                    file2 = files[i + 1] if i + 1 < len(files) else None

                    pcd_path = os.path.join(root, file2) if file2 else None
                    source_path = pcd_path.replace("\\", "/")
                    root = root.replace("\\", "/")
                    transformed_cloud = perform_icp(source_path, target_path)
                    transformed_points = np.asarray(transformed_cloud.points)
                    # join all the frames point cloud
                    condition1 = np.logical_and(transformed_points[:, 1] <= 7.5, transformed_points[:, 1] >= -7.8)
                    points = transformed_points[condition1]
                    condition2 = np.logical_and(points[:, 2] <= 3, points[:, 2] >= 1)
                    transformed_points = points[condition2]
                    target_points = np.vstack((transformed_points, target_points))

                targetPoints = o3d.geometry.PointCloud()
                targetPoints.points = o3d.utility.Vector3dVector(target_points)
                vis = o3d.visualization.Visualizer()
                # 创建窗口，设置窗口标题
                vis.create_window(window_name="point_cloud")
                # 设置点云渲染参数
                opt = vis.get_render_option()
                # 设置背景色（这里为白色）
                opt.background_color = np.array([255, 255, 255])
                # 设置渲染点的大小
                opt.point_size = 3.0
                vis.add_geometry(targetPoints)
                vis.run()
