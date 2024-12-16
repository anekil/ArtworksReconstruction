import io

import numpy as np
import torch
import random

from PIL import Image


class InpaintingDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def apply_damage(self, image):
        img = np.array(image)  # Convert PIL Image to NumPy array
        height, width, _ = img.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        mask_size = int(min(height, width) // 4)
        top_left_x = random.randint(0, width - mask_size)
        top_left_y = random.randint(0, height - mask_size)

        mask[top_left_y:top_left_y + mask_size, top_left_x:top_left_x + mask_size] = 1
        img[top_left_y:top_left_y + mask_size, top_left_x:top_left_x + mask_size] = [255, 255, 255]

        return Image.fromarray(img), Image.fromarray(mask * 255)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if isinstance(row['image'], dict):
            image = row['image']['bytes']
            image = Image.open(io.BytesIO(image))
        elif isinstance(row['image'], bytes):
            image = Image.open(io.BytesIO(row['image']))
        else:
            image = row['image']

        if image.mode != 'RGB':
            image = image.convert('RGB')

        damaged_image, mask = self.apply_damage(image)

        if self.transform:
            original_image = self.transform(image)
            damaged_image = self.transform(damaged_image)
            mask = self.transform(mask).squeeze(0)
            damaged_image = torch.cat([damaged_image, mask.unsqueeze(0)], dim=0)
        else:
            original_image = image

        return damaged_image, original_image

    def __len__(self):
        return len(self.df)