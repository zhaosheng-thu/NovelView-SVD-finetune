from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from diffusers.models import AutoencoderKLTemporalDecoder # transformers==4.38.2 tokenizers==0.15.2 
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import cv2
from rembg import remove
import numpy as np
from PIL import Image
import os
import json
        
def image_loader(image_name):
    image = Image.open(image_name)
    image = transforms.ToTensor()(image).unsqueeze(0) # 给张量增添一个第零维度
    return image

def load_image_sv3d(input_img_path, image_frame_ratio=None):
    image = Image.open(input_img_path)
    # print("image_mode", image.mode)
    if image.mode == "RGBA":
        pass
    else:
        # remove bg
        image.thumbnail([768, 768], Image.Resampling.LANCZOS)
        image = remove(image.convert("RGBA"), alpha_matting=True)
    # resize object in frame
    image_arr = np.array(image)
    in_w, in_h = image_arr.shape[:2]
    ret, mask = cv2.threshold(
        np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY
    )
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    side_len = (
        int(max_size / image_frame_ratio)
        if image_frame_ratio is not None
        else in_w
    )
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded_image[
        center - h // 2 : center - h // 2 + h,
        center - w // 2 : center - w // 2 + w,
    ] = image_arr[y : y + h, x : x + w]
    # resize frame to 512*512？256*256？576*576？
    # TODO: the size should be aligned with the size in the dataset and the config file
    rgba = Image.fromarray(padded_image).resize((576, 576), Image.LANCZOS)
    # white bg
    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    
    image = ToTensor()(input_image)
    image = image * 2.0 - 1.0
    # device = "cuda" # TODO: modify this if not using GPU
    image = image.unsqueeze(0)
    H, W = image.shape[2:]
    assert image.shape[1] == 3
    F = 8
    C = 4
    f = 7 # TODO check f, but here f = 1
    shape = (f, C, H // F, W // F)
    
    return image, shape


def latent_image2pil_image(latent_image):
    # normalize the image pixel from [-1, 1] to [0, 1]
    latent_image = (latent_image / 2 + 0.5).clamp(0, 1)
    latent_image = latent_image.detach().cpu().permute(0, 2, 3, 1).numpy()
    latent_image = latent_image * 255
    latent_image = latent_image.round().astype("uint8")
    pil_image = Image.fromarray(latent_image[0])
    return pil_image
     
     
vae = AutoencoderKLTemporalDecoder.from_pretrained("/root/zyma/szhao-06/model-weights/svd/vae").to("cuda")
path = "/root/zyma/szhao-06/generative-models/objaverse-rendering-video/valid_paths_video.json"
with open(path, 'r') as f:
    paths = json.load(f)
    
# list 
for item in paths:
    index_item = item.split("/")[-1][:-4]
    filename = os.path.join(path.split("/valid_paths_video")[0], index_item)
    print("index", index_item, filename)
    for idx in range(21):  
        image, _ = load_image_sv3d(os.path.join(filename, '%03d.png' % idx))
        image = image.to("cuda")
        latent_dist = vae.encode(image).latent_dist
        latents_sp = 0.18215 * latent_dist.sample() #s ample mean + std.randn
        latents = 0.18215 * latent_dist.mode() # mean
        torch.save(latents_sp, os.path.join(filename, '%03d.pt' % idx))
        
# image = image_loader(IMG_PATH) if IMG_PATH is not None else torch.rand(1, 3, 512, 512) * 2 - 1 