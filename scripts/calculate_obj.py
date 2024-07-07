import os
import json

def count_subdirectories(root_dir):
    # 检查路径是否存在并且是一个目录
    if not os.path.exists(root_dir) or not os.path.isdir(root_dir):
        raise ValueError(f"'{root_dir}' is not a valid directory path.")
    
    # 获取root_dir目录下的所有子目录
    subdirs = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
    
    # 返回子目录的数量
    return len(subdirs)


def append_to_json_file(file_path, new_data):
    # Check if the file exists
    
    if os.path.exists(file_path):
        # Read the existing data
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    # Append the new data and remove duplicates
    data_set = set(data)
    new_data_set = set(new_data)
    combined_data_set = data_set.union(new_data_set)

    # Convert the set back to a list
    combined_data = list(combined_data_set)

    # Write the updated data back to the file
    with open(file_path, "w") as f:
        json.dump(combined_data, f, indent=2)

def count_folders_with_42_files(root_directory, json_file_path):
    count = 0
    new_data = []
    for root, dirs, files in os.walk(root_directory):
        for dir in dirs:
            subfolder_path = os.path.join(root, dir)
            try:
                if len([name for name in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, name))]) == 42:
                    count += 1
                    pth = subfolder_path.split("objaverse-rendering")[-1] + ".glb"
                    new_data.append(pth)
            except PermissionError:
                # 忽略因权限问题无法访问的文件夹
                continue
    append_to_json_file(json_file_path, new_data)
    return count

# 示例用法
root_directory = "/root/szhao/zero123/objaverse-rendering/views_video_576"  # 将此路径替换为实际的根目录路径
try:
    num_folders = count_folders_with_42_files(root_directory, json_file_path="/root/szhao/zero123/objaverse-rendering/views_video_576/valid_paths_video_576.json")
    print(f"Number of folders with exactly 42 files in '{root_directory}': {num_folders}")
except Exception as e:
    print(e)

# 示例用法
# root_directory = "/root/szhao/zero123/objaverse-rendering/views_new_video"  # 将此路径替换为实际的根目录路径
# try:
#     num_subdirectories = count_subdirectories(root_directory)
#     print(f"Number of subdirectories in '{root_directory}': {num_subdirectories}")
# except ValueError as e:
#     print(e)
