import io

import cv2
import numpy as np
import torch
import random

from PIL import Image, ImageDraw


class InpaintingDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def apply_damage(self, image):
        img = np.array(image)
        height, width, _ = img.shape
        mask = np.ones((height, width), dtype=np.uint8) * 255  # Initialize white mask

        num_blobs = random.randint(3, 6)  # Number of blobs
        max_radius = min(height, width) // 8  # Maximum radius for blobs

        for _ in range(num_blobs):
            radius = random.randint(max_radius // 4, max_radius)
            center_x = random.randint(radius, width - radius)
            center_y = random.randint(radius, height - radius)

            # Draw a filled circle on the mask (black) and image (white damage)
            cv2.circle(mask, (center_x, center_y), radius, 0, -1)  # Black blob on mask
            cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), -1)  # White blob on image

        return Image.fromarray(img), Image.fromarray(mask)

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