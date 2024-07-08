import json
import multiprocessing
import os
import torch
from typing import Optional
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from diffusers.models import AutoencoderKLTemporalDecoder
import torchvision.transforms as transforms
from PIL import Image
from rembg import remove
import cv2
import numpy as np
import subprocess

def image_loader(image_name):
    image = Image.open(image_name)
    image = transforms.ToTensor()(image).unsqueeze(0)
    return image

def load_image_sv3d(input_img_path, image_frame_ratio=None):
    image = Image.open(input_img_path)
    if image.mode != "RGBA":
        image.thumbnail([768, 768], Image.Resampling.LANCZOS)
        image = remove(image.convert("RGBA"), alpha_matting=True)
    
    image_arr = np.array(image)
    in_w, in_h = image_arr.shape[:2]
    ret, mask = cv2.threshold(np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    side_len = int(max_size / image_frame_ratio) if image_frame_ratio else in_w
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded_image[center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w] = image_arr[y : y + h, x : x + w]
    rgba = Image.fromarray(padded_image).resize((576, 576), Image.LANCZOS) # TODO: Here is the size
    
    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    
    image = transforms.ToTensor()(input_image)
    image = image * 2.0 - 1.0
    image = image.unsqueeze(0)
    return image, image.shape

def worker(
    queue: multiprocessing.Queue,
    count: multiprocessing.Value,
    gpu: int,
    vae_model_path: str,
    size: int,
) -> None:

    # vae = AutoencoderKLTemporalDecoder.from_pretrained(vae_model_path).to(f"cuda:{gpu}")
    while True:
        item = queue.get()
        if item is None:
            break
        
        filename, idx = item
        print(f"filename, {filename}, {idx} gpu, {gpu}")
        command = (
            f" CUDA_VISIBLE_DEVICES={gpu} "
            f" python tests/latent/vae.py --"
            f" --filename {filename}"
            f" --idx {idx}"
            f" --size {size}"
        )
        
        # print(command)
        subprocess.run(command, shell=True)
        
        with count.get_lock():
            count.value += 1

        queue.task_done()

if __name__ == "__main__":
    # try:
    #     multiprocessing.set_start_method('spawn')  # Set the start method to 'spawn'
    # except RuntimeError:
    #     pass
    
    workers_per_gpu = 12
    num_gpus = 10
    size = 256
    json_paths = "/root/szhao/zero123/objaverse-rendering/views_video_576/valid_paths_video_576.json"
    vae_model_path = "/root/szhao/model-weights/stable-video-diffusion-img2vid-xt/vae"
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)
    
    for gpu_i in range(num_gpus):
        for worker_i in range(workers_per_gpu):
            worker_i = gpu_i * workers_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i, vae_model_path, size)
            )
            process.daemon = True
            process.start()
    
    with open(json_paths, "r") as f:
        paths = json.load(f)
    
    for item in paths:
        index_item = item.split("/")[-1][:-4]
        filename = os.path.join(json_paths.split("/valid_paths")[0], index_item)
        print("index", index_item, filename)
        for idx in range(21):
            queue.put((filename, idx))
    
    queue.join()
    
    for i in range(num_gpus * workers_per_gpu):
        queue.put(None)
