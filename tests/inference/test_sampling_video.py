import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from rembg import remove
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor


def sample(
    input_path: str = "outputs/simple_video_sample/sv3d_u/000000.jpg",  # Can either be image file or folder with image files
    num_frames: Optional[int] = None,  # 21 for SV3D
    num_steps: Optional[int] = None,
    version: str = "sv3d_u",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 6,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    elevations_deg: Optional[float | List[float]] = [20,15,10,10,0,0,0],  # For SV3D
    azimuths_deg: Optional[List[float]] = [1,50,120,170,240,310,350],  # For SV3D
    image_frame_ratio: Optional[float] = None,
    verbose: Optional[bool] = False,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd/")
        model_config = "scripts/sampling/configs/svd.yaml"
    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    elif version == "svd_image_decoder":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(
            output_folder, "outputs/simple_video_sample/svd_image_decoder/"
        )
        model_config = "scripts/sampling/configs/svd_image_decoder.yaml"
    elif version == "svd_xt_image_decoder":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(
            output_folder, "outputs/simple_video_sample/svd_xt_image_decoder/"
        )
        model_config = "scripts/sampling/configs/svd_xt_image_decoder.yaml"
    elif version == "sv3d_u":
        num_frames = 21
        num_steps = default(num_steps, 50)
        output_folder = default(output_folder, "outputs/simple_video_sample/sv3d_u/")
        model_config = "scripts/sampling/configs/sv3d_u.yaml"
        cond_aug = 1e-5
    elif version == "sv3d_p":
        num_frames = 21
        num_steps = default(num_steps, 50)
        output_folder = default(output_folder, "outputs/simple_video_sample/sv3d_p/")
        model_config = "/root/zyma/szhao-06/generative-models/scripts/sampling/configs/sv3d_p.yaml"
        cond_aug = 1e-5
        if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
            elevations_deg = [elevations_deg] * num_frames
        assert (
            len(elevations_deg) == num_frames
        ), f"Please provide 1 value, or a list of {num_frames} values for elevations_deg! Given {len(elevations_deg)}"
        polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
        if azimuths_deg is None:
            azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
        assert (
            len(azimuths_deg) == num_frames
        ), f"Please provide a list of {num_frames} values for azimuths_deg! Given {len(azimuths_deg)}"
        azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
        azimuths_rad[:-1].sort()
    else:
        raise ValueError(f"Version {version} does not exist.")

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    
    state_dict_keys = []
    state_dict_vals = []
    # print("model", model.first_stage_model)
    # for k in model.state_dict().keys():
    #     if "diffusion_model.out." in k:
    #         state_dict_keys.append(k)
    #         state_dict_vals.append(model.state_dict()[k])
    #         print("k", k, model.state_dict()[k].shape)
    # print("/nYes")if isinstance(model.first_stage_model.decoder, VideoDecoder) else print("No")

    torch.manual_seed(seed)

    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError

    for input_img_path in all_img_paths:
        if "sv3d" in version:
            image = Image.open(input_img_path)
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
            # resize frame to 576x576
            rgba = Image.fromarray(padded_image).resize((256, 256), Image.LANCZOS)
            # white bg
            rgba_arr = np.array(rgba) / 255.0
            rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
            input_image = Image.fromarray((rgb * 255).astype(np.uint8))

        else:
            with Image.open(input_img_path) as image:
                if image.mode == "RGBA":
                    input_image = image.convert("RGB")
                w, h = image.size

                if h % 64 != 0 or w % 64 != 0:
                    width, height = map(lambda x: x - x % 64, (w, h))
                    input_image = input_image.resize((width, height))
                    print(
                        f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                    )

        image = ToTensor()(input_image)
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024) and "sv3d" not in version:
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if (H, W) != (576, 576) and "sv3d" in version:
            print(
                "WARNING: The conditioning frame you provided is not 576x576. This leads to suboptimal performance as model was only trained on 576x576."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["cond_frames_without_noise"] = image
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        if "sv3d_p" in version:
            value_dict["polars_rad"] = polars_rad
            value_dict["azimuths_rad"] = azimuths_rad
            
        ############################ Here ############################
        # latent_aug = torch.load("/root/zyma/szhao-06/datasets/svd-sv3d/latents_aug.pt")
        latents_wo = torch.load("/root/zyma/szhao-06/datasets/svd-sv3d/latents.pt")
        print("latents_wo", latents_wo.shape)
        model.en_and_decode_n_samples_a_time = decoding_t
        sample_x = model.decode_first_stage(latents_wo)
        sample_x = (sample_x / 2 + 0.5).clamp(0, 1)
        sample_x = sample_x.detach().cpu().permute(0, 2, 3, 1).numpy()
        sample_x = sample_x * 255
        sample_x = sample_x.round().astype("uint8")
        pil_image = Image.fromarray(sample_x[0])
        pil_image.save("/root/zyma/szhao-06/datasets/svd-sv3d/test_wo.png")


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter


if __name__ == "__main__":
    Fire(sample)
