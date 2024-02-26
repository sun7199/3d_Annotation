from singleFrame import singleFrame


if __name__ == "__main__":
    folder_path = "scene_001/002"
    for root, dirs, files in os.walk(folder_path):
        if files:
            if files[0].endswith("json"):
                files = sorted(files)
                for i in range(0, len(files), 2):
                    file1 = files[i]
                    file2 = files[i + 1] if i + 1 < len(files) else None

                    json_path = os.path.join(root, file1)
                    json_path = json_path.replace("\\", "/")
                    pcd_path = os.path.join(root, file2) if file2 else None
                    pcd_path = pcd_path.replace("\\", "/")
                    root = root.replace("\\", "/")
                    singleFrame(pcd_path, json_path, root)
