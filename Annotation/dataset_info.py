import os
import json
from openpyxl import Workbook


def read_json(filepath):
    with open(filepath, "rb") as file:
        json_data = json.load(file)
    return json_data


def get_jsonInfo(num_of_objects, numEachType, types, jsonPath):
    jsonfile = read_json(jsonPath)
    for obj in jsonfile["objects"]:
        num_of_objects += 1
        # 解析每个目标的数据
        shapeType = obj["shapeType"]
        for i in range(0, len(types)):
            if types[i] == shapeType:
                numEachType[i] += 1
            else:
                types.append(shapeType)
                numEachType.append(1)

        print(num_of_objects, types, numEachType)
        return num_of_objects, types, numEachType


if __name__ == "__main__":
    folder_path = "E:/AZreport/AZreport"
    types = []

    # Create a new workbook
    workbook = Workbook()

    # Select the active worksheet
    worksheet = workbook.active

    for root, dirs, files in os.walk(folder_path):
        if files:
            if files[0].endswith("json"):
                # Split the path into directory and filename
                directory, filename = os.path.split(root)

                # Collect parts of the directory path
                parts = directory.split("/")  # os.sep represents the separator used in the current OS
                # Create a new path with only specific parts
                new_directory = os.path.join(*parts[2:])
                files = sorted(files)
                numEachType = [0 for i in types]
                num_of_objects = 0
                for i in range(0, len(files), 2):
                    file1 = files[i]
                    file2 = files[i + 1] if i + 1 < len(files) else None
                    json_path = os.path.join(root, file1)
                    json_path = json_path.replace("\\", "/")
                    pcd_path = os.path.join(root, file2) if file2 else None
                    pcd_path = pcd_path.replace("\\", "/")
                    root = root.replace("\\", "/")
                    jsonfile = read_json(json_path)
                    for obj in jsonfile["objects"]:
                        num_of_objects += 1
                        found = False
                        # 解析每个目标的数据
                        shapeType = obj["label"]
                        for i in range(0, len(types)):
                            if types[i] == str(shapeType):
                                numEachType[i] += 1
                                found = True
                                break
                        if not found:
                            types.append(shapeType)
                            numEachType.append(1)

                print(num_of_objects, types, numEachType)
                row = [new_directory]
                row.extend([num_of_objects])
                row.extend(numEachType)
                print(row)
                worksheet.append(row)

    first_row = ["pathname", "total objects"]
    first_row.extend(types)
    worksheet.append(first_row)
    workbook.save('数据统计.xlsx')
