from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from torchvision.transforms import ToTensor
import torchvision
from einops import rearrange
from sgm.util import instantiate_from_config
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler
from rembg import remove


# Some hacky things to make experimentation easier
def make_transform_multi_folder_data(paths, caption_files=None, **kwargs):
    ds = make_multi_folder_data(paths, caption_files, **kwargs)
    return TransformDataset(ds)

def make_nfp_data(base_path):
    dirs = list(Path(base_path).glob("*/"))
    print(f"Found {len(dirs)} folders")
    print(dirs)
    tforms = [transforms.Resize(512), transforms.CenterCrop(512)]
    datasets = [NfpDataset(x, image_transforms=copy.copy(tforms), default_caption="A view from a train window") for x in dirs]
    return torch.utils.data.ConcatDataset(datasets)


class VideoDataset(Dataset):
    def __init__(self, root_dir, image_transforms, caption_file, offset=8, n=2):
        self.root_dir = Path(root_dir)
        self.caption_file = caption_file
        self.n = n
        ext = "mp4"
        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.offset = offset

        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms
        with open(self.caption_file) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        self.captions = dict(rows)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        for i in range(10):
            try:
                return self._load_sample(index)
            except Exception:
                # Not really good enough but...
                print("uh oh")

    def _load_sample(self, index):
        n = self.n
        filename = self.paths[index]
        min_frame = 2*self.offset + 2
        vid = cv2.VideoCapture(str(filename))
        max_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame_n = random.randint(min_frame, max_frames)
        vid.set(cv2.CAP_PROP_POS_FRAMES,curr_frame_n)
        _, curr_frame = vid.read()

        prev_frames = []
        for i in range(n):
            prev_frame_n = curr_frame_n - (i+1)*self.offset
            vid.set(cv2.CAP_PROP_POS_FRAMES,prev_frame_n)
            _, prev_frame = vid.read()
            prev_frame = self.tform(Image.fromarray(prev_frame[...,::-1]))
            prev_frames.append(prev_frame)

        vid.release()
        caption = self.captions[filename.name]
        data = {
            "image": self.tform(Image.fromarray(curr_frame[...,::-1])),
            "prev": torch.cat(prev_frames, dim=-1),
            "txt": caption
        }
        return data

# end hacky things


def make_tranforms(image_transforms):
    # if isinstance(image_transforms, ListConfig):
    #     image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms = []
    image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    image_transforms = transforms.Compose(image_transforms)
    return image_transforms


def make_multi_folder_data(paths, caption_files=None, **kwargs):
    """Make a concat dataset from multiple folders
    Don't suport captions yet

    If paths is a list, that's ok, if it's a Dict interpret it as:
    k=folder v=n_times to repeat that
    """
    list_of_paths = []
    if isinstance(paths, (Dict, DictConfig)):
        assert caption_files is None, \
            "Caption files not yet supported for repeats"
        for folder_path, repeats in paths.items():
            list_of_paths.extend([folder_path]*repeats)
        paths = list_of_paths

    if caption_files is not None:
        datasets = [FolderData(p, caption_file=c, **kwargs) for (p, c) in zip(paths, caption_files)]
    else:
        datasets = [FolderData(p, **kwargs) for p in paths]
    return torch.utils.data.ConcatDataset(datasets)



class NfpDataset(Dataset):
    def __init__(self,
        root_dir,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        ) -> None:
        """assume sequential frames and a deterministic transform"""

        self.root_dir = Path(root_dir)
        self.default_caption = default_caption

        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        return len(self.paths) - 1


    def __getitem__(self, index):
        prev = self.paths[index]
        curr = self.paths[index+1]
        data = {}
        data["image"] = self._load_im(curr)
        data["prev"] = self._load_im(prev)
        data["txt"] = self.default_caption
        return data

    def _load_im(self, filename):
        im = Image.open(filename).convert("RGB")
        return self.tform(im)

class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(transform_fn)])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def transform_fn(x):
        return rearrange(x * 2. - 1., 'c h w -> h w c')

    def train_dataloader(self):
        dataset = NVObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = NVObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(NVObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='', # TODO: modify this path
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,
        validation=False
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        with open(os.path.join(root_dir, 'valid_paths.json')) as f:
            self.paths = json.load(f)
           
        for i in range(len(self.paths)):
            self.paths[i] = self.paths[i].split("/")[-1][:-4]
            
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T
        
    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):

        data = {}
        total_view = self.total_view
        index_target, index_cond = random.sample(range(total_view), 2) # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        print(f"\nindex: {index}, {self.paths[index]}")

        if self.return_paths:
            data["path"] = str(filename)
        
        color = [1., 1., 1., 1.]

        try:
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
        except:
            # very hacky solution, sorry about this
            filename = os.path.join(self.root_dir, '0a8c36767de249e89fe822f48249c10c') # this one we know is valid
            target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
            cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            target_im = torch.zeros_like(target_im)
            cond_im = torch.zeros_like(cond_im)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


class NVObjaverseData(ObjaverseData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def load_image_nv(self, input_img_path, image_frame_ratio=None):
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
        rgba = Image.fromarray(padded_image).resize((256, 256), Image.LANCZOS)
        # white bg
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
        f = 21 # TODO check f, but here f = 1
        shape = (f, C, H // F, W // F)
        
        return image, shape
    
        
    def get_azimuth_theta_z(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = np.mod(azimuth_target - azimuth_cond, 2 * math.pi)
        d_z = z_target - z_cond
        
        return np.array([d_azimuth, d_theta, d_z])
    
    
    def __getitem__(self, index):
        data = {}
        total_view = self.total_view
        index_target, index_cond = random.sample(range(total_view), 2) # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        print(f"\nindex in simple.py: {index}, {self.paths[index]}")

        if self.return_paths:
            data["path"] = str(filename)
        
        try:
            target_im, shape = self.load_image_nv(os.path.join(filename, '%03d.png' % index_target))
            # target_im = torch.load(os.path.join(filename, '%03d.pt' % index_target))
            cond_im, shape = self.load_image_nv(os.path.join(filename, '%03d.png' % index_cond))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
        except:
            # very hacky solution
            filename = os.path.join(self.root_dir, '0a8c36767de249e89fe822f48249c10c') # this one we know is valid
            target_im, shape = self.load_image_nv(os.path.join(filename, '%03d.png' % index_target))
            # target_im = torch.load(os.path.join(filename, '%03d.pt' % index_target))
            cond_im, shape = self.load_image_nv(os.path.join(filename, '%03d.png' % index_cond))
            target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            target_im = torch.zeros_like(target_im)
            cond_im = torch.zeros_like(cond_im)

        cond_aug = 1e-5
        data["target_frames_without_noise"] = target_im
        data["cond_frames_without_noise"] = cond_im
        data["cond_aug"] = torch.tensor(cond_aug)
        data["cond_frames"] = cond_im + cond_aug * torch.randn_like(cond_im)
        d_azimuth, d_theta, d_z = self.get_azimuth_theta_z(target_RT, cond_RT)
        
        f = shape[0]
        d_azimuth = torch.tensor(d_azimuth[0]).repeat(f)
        d_theta = torch.tensor(np.deg2rad(90 - math.degrees(-d_theta[0]))).repeat(f) # pay atten to the '-' sign here
        d_z = torch.tensor(d_z[0]).repeat(f)
        # print("d_azimuth in simple.py: ", d_azimuth.shape, d_z.shape, d_theta.shape)
        data['cond_aug'] = data['cond_aug'].repeat(f)
        """
        because for the conditions of the azimuths, polars and heights, they are the same, and for each depth of Unet 3D
        we need to repeat the same condition for f times, here f indicates the depth of Unet 3D
        """
        data["azimuths_rad"] = d_azimuth
        data["polars_rad"] = d_theta
        data["height_z"] = d_z
        data["num_video_frames"] = f # this is not a tensor
        
        if self.postprocess is not None:
            data = self.postprocess(data)
            
        return data
    
def transform_fn(x):
    return rearrange(x * 2. - 1., 'c h w -> h w c')

class SV3DObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(transform_fn)])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)
    
    def train_dataloader(self):
        dataset = SV3DObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = SV3DObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(SV3DObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)    

# class for sv3d
class SV3DObjaverseData(ObjaverseData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def load_image_sv3d(self, input_img_path, image_frame_ratio=None):
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
        # TODO: resize frame to 512*512？256*256？576*576？
        rgba = Image.fromarray(padded_image).resize((576, 576), Image.LANCZOS)
        # white bg
        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
        input_image = Image.fromarray((rgb * 255).astype(np.uint8))
        
        image = ToTensor()(input_image)
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0) # first dimension is the batch size
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        f = 21
        shape = (f, C, H // F, W // F)
        
        return image, shape
    
        
    def get_azimuth_theta_z(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = np.mod(azimuth_target - azimuth_cond, 2 * math.pi)
        d_z = z_target - z_cond
        
        return np.array([d_azimuth, d_theta, d_z])
    
    
    def __getitem__(self, index):
        data = {}
        total_view = self.total_view
        filename = os.path.join(self.root_dir, self.paths[index])

        print(f"\nindex in simple.py: {index}, {self.paths[index]}")

        if self.return_paths:
            data["path"] = str(filename)
            
        index_cond = 0
        
        # preprocess the first frame as condition, reading the png
        try:
            cond_im, _ = self.load_image_sv3d(os.path.join(filename, '%03d.png' % index_cond))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
        except:
            filename = os.path.join(self.root_dir, '0a8c36767de249e89fe822f48249c10c') # this one we know is valid
            cond_im, _ = self.load_image_sv3d(os.path.join(filename, '%03d.png' % index_cond))
            cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
            cond_im = torch.zeros_like(cond_im)
            print("load except object")
            
        target_frame_wo_noise = [] # prepraing for the target frames without noise, all the latents encoded by vae, ended with .pt
        azimuth = []
        polar = []
        height = []
        for index_target in range(total_view):
            try:
                target_im = torch.load(os.path.join(filename, '%03d.pt' % index_target))
                target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
                print("target_im in simple.py: ", target_im.shape)
                
            except:
                # very hacky solution
                filename = os.path.join(self.root_dir, '0a8c36767de249e89fe822f48249c10c') # this one we know is valid
                target_im = torch.load(os.path.join(filename, '%03d.pt' % index_target))
                target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
                target_im = torch.zeros_like(target_im)

            d_azimuth, d_theta, d_z = self.get_azimuth_theta_z(target_RT, cond_RT)
            
            # trans to deg 
            target_frame_wo_noise.append(target_im)
            azimuth.append(math.degrees(d_azimuth[0]))
            polar.append(math.degrees(-d_theta[0]))
            height.append(d_z[0])
        
        # azimuth = [x + random.uniform(-5, 5) for x in azimuth]
        # azimuth = azimuth[1:] + azimuth[:1] # Here change the azimuth degree to 360, which is the same as the last frame 
        # process the azimuth, polar and height
        polars_rad = [np.deg2rad(90 - e) for e in polar]
        azimuths_rad = [np.deg2rad((a - azimuth[0]) % 360) for a in azimuth]
        azimuths_rad[0:].sort()
        
        cond_aug = 1e-5
        data["target_frames_without_noise"] = torch.cat(target_frame_wo_noise, dim=0)
        data["cond_frames_without_noise"] = cond_im
        data["cond_aug"] = torch.tensor(cond_aug).repeat(total_view)
        data["cond_frames"] = cond_im + cond_aug * torch.randn_like(cond_im)
        data["azimuths_rad"] = torch.tensor(azimuths_rad)
        data["polars_rad"] = torch.tensor(polars_rad)
        data["height_z"] = torch.tensor(height)
        print("data['azi'] in simple.py: ", data["azimuths_rad"].shape)
        
        data["num_video_frames"] = total_view # this is not a tensor
        
        if self.postprocess is not None:
            data = self.postprocess(data)
            
        for k, v in data.items():
            print("in simple.py", k, v.shape) if isinstance(v, torch.Tensor) else print(k, v)
        return data  


class FolderData(Dataset):
    def __init__(self,
        root_dir,
        caption_file=None,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        postprocess=None,
        return_paths=False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_caption = default_caption
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                ext = Path(caption_file).suffix.lower()
                if ext == ".json":
                    captions = json.load(f)
                elif ext == ".jsonl":
                    lines = f.readlines()
                    lines = [json.loads(x) for x in lines]
                    captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
                else:
                    raise ValueError(f"Unrecognised format: {ext}")
            self.captions = captions
        else:
            self.captions = None

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # Only used if there is no caption file
        self.paths = []
        for e in ext:
            self.paths.extend(sorted(list(self.root_dir.rglob(f"*.{e}"))))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        if self.captions is not None:
            return len(self.captions.keys())
        else:
            return len(self.paths)

    def __getitem__(self, index):
        data = {}
        if self.captions is not None:
            chosen = list(self.captions.keys())[index]
            caption = self.captions.get(chosen, None)
            if caption is None:
                caption = self.default_caption
            filename = self.root_dir/chosen
        else:
            filename = self.paths[index]

        if self.return_paths:
            data["path"] = str(filename)

        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        data["image"] = im

        if self.captions is not None:
            data["txt"] = caption
        else:
            data["txt"] = self.default_caption

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
import random

class TransformDataset():
    def __init__(self, ds, extra_label="sksbspic"):
        self.ds = ds
        self.extra_label = extra_label
        self.transforms = {
            "align": transforms.Resize(768),
            "centerzoom": transforms.CenterCrop(768),
            "randzoom": transforms.RandomCrop(768),
        }


    def __getitem__(self, index):
        data = self.ds[index]

        im = data['image']
        im = im.permute(2,0,1)
        # In case data is smaller than expected
        im = transforms.Resize(1024)(im)

        tform_name = random.choice(list(self.transforms.keys()))
        im = self.transforms[tform_name](im)

        im = im.permute(1,2,0)

        data['image'] = im
        data['txt'] = data['txt'] + f" {self.extra_label} {tform_name}"

        return data

    def __len__(self):
        return len(self.ds)

def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split='train',
    image_key='image',
    caption_key='txt',
    ):
    """Make huggingface dataset with appropriate list of transforms applied
    """
    ds = load_dataset(name, split=split)
    tform = make_tranforms(image_transforms)

    assert image_column in ds.column_names, f"Didn't find column {image_column} in {ds.column_names}"
    assert text_column in ds.column_names, f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds

class TextOnly(Dataset):
    def __init__(self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]



import random
import json
class IdRetreivalDataset(FolderData):
    def __init__(self, ret_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(ret_file, "rt") as f:
            self.ret = json.load(f)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        key = self.paths[index].name
        matches = self.ret[key]
        if len(matches) > 0:
            retreived = random.choice(matches)
        else:
            retreived = key
        filename = self.root_dir/retreived
        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        # data["match"] = im
        data["match"] = torch.cat((data["image"], im), dim=-1)
        return data
