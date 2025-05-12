import glob
import math
import numpy as np
import os
import random
import rasterio
import torch
import torchvision.transforms.functional as F
import yaml

from rasterio.windows import Window
from torch.utils.data import Dataset



class PVDataset(Dataset):
    def __init__(self, data_path, stats_file, split="train_samples", image_size=224, key="bg"):
        self.split = split
        self.image_paths = sorted(glob.glob(os.path.join(data_path, split, "images", f"*{key}.tif")))
        self.mask_paths = sorted(glob.glob(os.path.join(data_path, split, "masks", f"*{key}.tif")))
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch between images and masks"
        self.image_size = image_size

        with open(stats_file, "r") as f:
            stats = yaml.safe_load(f)
            self.mean = torch.tensor(stats["mean"]).view(-1, 1, 1)
            self.std = torch.tensor(stats["stddev"]).view(-1, 1, 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.image_paths[idx]) as img_src:
            full_height, full_width = img_src.height, img_src.width
            crop_h, crop_w = self.image_size, self.image_size

            if self.split == "test_samples":
                top = (full_height - crop_h) // 2
                left = (full_width - crop_w) // 2
            else:
                top = random.randint(0, full_height - crop_h)
                left = random.randint(0, full_width - crop_w)

            window = Window(left, top, crop_w, crop_h)
            image = img_src.read(window=window) 
            
        with rasterio.open(self.mask_paths[idx]) as mask_src:
            mask = mask_src.read(1, window=window)

        image = torch.tensor(image).float()
        mask = torch.tensor(mask).unsqueeze(0).float()
        
        if self.split == "train_samples":
            if random.random() < 0.50:
                image = F.hflip(image)
                mask = F.hflip(mask)
            if random.random() < 0.50:
                image = F.vflip(image)
                mask = F.vflip(mask)
            if random.random() < 0.75:
                angle = random.choice([90, 180, 270])
                image = F.rotate(image, angle)
                mask = F.rotate(mask, angle)

        image = (image - self.mean) / self.std

        return image, mask, idx
    

class InferenceDataset(Dataset):
    def __init__(self, tif_path, stats_file, image_size=224, min_overlap=0.1):
        self.tif_path = tif_path
        self.image_size = image_size
        assert 0 <= min_overlap < 1, "min_overlap_frac must be in [0, 1)"
        self.min_overlap = min_overlap
        self.windows = self.get_windows_from_tif()
        print(len(self.windows))

        with open(stats_file, "r") as f:
            stats = yaml.safe_load(f)
            self.mean = torch.tensor(stats["mean"]).view(-1, 1, 1)
            self.std = torch.tensor(stats["stddev"]).view(-1, 1, 1)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        with rasterio.open(self.tif_path) as img_src:
            image = img_src.read(window=window) 

        image = torch.tensor(image).float()
        image = (image - self.mean) / self.std

        return image, idx
    
    def get_windows_from_tif(self):
        with rasterio.open(self.tif_path) as src:
            H, W = src.height, src.width
            
            min_ov_h = int(self.min_overlap * self.image_size)
            min_ov_w = int(self.min_overlap * self.image_size)

            max_stride_h = self.image_size - min_ov_h
            max_stride_w = self.image_size - min_ov_w

            n_h = math.ceil((H - self.image_size) / max_stride_h) + 1
            n_w = math.ceil((W - self.image_size) / max_stride_w) + 1

            stride_h = (H - self.image_size) // (n_h - 1) if n_h > 1 else 0
            stride_w = (W - self.image_size) // (n_w - 1) if n_w > 1 else 0

            windows = []
            for i in range(n_h):
                for j in range(n_w):
                    row_off = min(i * stride_h, H - self.image_size)
                    col_off = min(j * stride_w, W - self.image_size)
                    window = Window(col_off, row_off, self.image_size, self.image_size)
                    windows.append(window)

            return windows
        