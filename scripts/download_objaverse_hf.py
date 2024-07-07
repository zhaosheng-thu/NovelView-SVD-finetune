import json
import random
import os
from dataclasses import dataclass
import gzip
import time

import boto3
import objaverse
import tyro
from tqdm import tqdm
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download, login
login_status = login(token="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")


@dataclass
class Args:
    start_i: int
    """total number of files uploaded"""

    end_i: int
    """total number of files uploaded"""

    skip_completed: bool = False
    """whether to skip the files that have already been downloaded"""


def get_completed_uids():
    # get all the files in the objaverse-images bucket
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("objaverse-images")
    bucket_files = [obj.key for obj in tqdm(bucket.objects.all())]

    dir_counts = {}
    for file in bucket_files:
        d = file.split("/")[0]
        dir_counts[d] = dir_counts.get(d, 0) + 1

    # get the directories with 12 files
    dirs = [d for d, c in dir_counts.items() if c == 12]
    return set(dirs)


def get_completed_uids_json():
    with open("input_models_path.json", "r") as f:
        data = json.load(f)
    list_uids = []
    for path in data:
        list_uids.append(path.split("/")[-1][:-4])
    return list_uids
    

def download_file(repo_id, file_path, local_dir="downloaded_files"):
    local_path = hf_hub_download(repo_id=repo_id, filename=file_path, repo_type="dataset", local_dir=local_dir)
    return local_path


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


def main():
    args = tyro.cli(Args)
    
    random.seed(42)

    # uids = objaverse.load_uids()
    with gzip.open("object-paths.json.gz", "rb") as f:
        object_paths = json.load(f)

    uids = list(object_paths.keys())
    print("num_uids", len(uids))
    
    random.shuffle(uids)

    # object_paths = objaverse._load_object_paths()
    uids = uids[args.start_i : args.end_i]
    print("uids", uids)

    # get the uids that have already been downloaded
    if not args.skip_completed:
        completed_uids = get_completed_uids_json()
        uids = [uid for uid in uids if uid not in completed_uids]

    repo_id = "allenai/objaverse"
    uid_object_paths = [
        object_paths[uid] for uid in uids
    ]
    
    print("uid_object_paths", uid_object_paths)
    print("\nDownloading files...")
    downloaded_files = []
    for file_path in tqdm(uid_object_paths):
        try:
            local_path = download_file(repo_id, file_path)
            downloaded_files.append(local_path)
        except Exception as e:
            print(f"Failed to download {file_path}: {e}")

    append_to_json_file("input_models_path.json", downloaded_files)
    # with open("input_models_path.json", "w") as f:
    #     json.dump(downloaded_files, f, indent=2)

# set the random seed to 42
if __name__ == "__main__":
    main()