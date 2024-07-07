import numpy as np
import math
from PIL import Image
from rembg import remove
import cv2
import os
from torchvision.transforms import ToTensor

def cartesian_to_spherical(xyz):
    xy = xyz[:,0]**2 + xyz[:,1]**2
    z = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2])  # Elevation angle from Z-axis down
    azimuth = np.arctan2(xyz[:,1], xyz[:,0])   # Azimuth angle in XY-plane from X-axis
    return np.array([theta, azimuth, z])

def get_azimuth_theta_z(target_RT, cond_RT):
    R, T = target_RT[:3, :3], target_RT[:, -1]
    T_target = -R.T @ T
    R, T = cond_RT[:3, :3], cond_RT[:, -1]
    T_cond = -R.T @ T
    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])
    
    d_theta = theta_target - theta_cond
    d_azimuth = np.mod(azimuth_target - azimuth_cond, 2 * math.pi)
    d_z = z_target - z_cond
    print(d_azimuth, d_theta, d_z)
    # 转换为角度值
    d_theta_deg = math.degrees(d_theta[0])
    d_azimuth_deg = math.degrees(d_azimuth[0])
    
    return np.array([d_azimuth_deg, d_theta_deg, d_z[0]])

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
    padded_image[center - h // 2:center - h // 2 + h, center - w // 2:center - w // 2 + w] = image_arr[y:y + h, x:x + w]
    rgba = Image.fromarray(padded_image).resize((256, 256), Image.LANCZOS)
    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    
    image = ToTensor()(input_image)
    image = image * 2.0 - 1.0
    image = image.unsqueeze(0)
    H, W = image.shape[2:]
    assert image.shape[1] == 3
    F = 8
    C = 4
    f = 21  # Check this value in your dataset/config
    
    return image, (f, C, H // F, W // F)

def load_img_npy():
    num_img = 21
    index_target = 3
    index_cond = 0
    filename = "/root/szhao/zero123/objaverse-rendering/views_new_video/0a8c36767de249e89fe822f48249c10c"
    
    target_im, shape = load_image_sv3d(os.path.join(filename, '%03d.png' % index_target))
    cond_im, _ = load_image_sv3d(os.path.join(filename, '%03d.png' % index_cond))
    target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
    cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
    
    d_azimuth, d_theta, d_z = get_azimuth_theta_z(target_RT, cond_RT)
    
    print("Delta Azimuth (degrees):", d_azimuth)
    print("Delta Theta (degrees):", d_theta)
    print("Delta Z:", d_z)
    
    elevations_deg = [20,15,10,10,0,0,0,5,5,0,0,0,-5,-10,-10,-10,-15,-20,-20,-20,-20]
    azimuths_deg = [0,20,35,50,60,75,90,110,125,140,160,180,200,210,220,240,260,280,300,330,340]
    polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
    azimuths_rad = [np.deg2rad((a - azimuths_deg[0]) % 360) for a in azimuths_deg]
    azimuths_rad[0:].sort()
    
    print(polars_rad)
    print(azimuths_rad)
    
    azimuths_deg = np.linspace(0, 360, 21 + 1)[:-1] % 360
    print(azimuths_deg)
    
    for i in range(21):
        azimuths_i = i / 21 * 360
        print(azimuths_i)

if __name__ == "__main__":  
    load_img_npy()
