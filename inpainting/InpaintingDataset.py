import io

import cv2
import numpy as np
import torch
import random

from PIL import Image


class InpaintingDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, to_input=None):
        self.df = df
        self.transform = transform
        self.to_input = to_input

    def apply_damage(self, image):
        img = np.array(image)
        height, width, _ = img.shape
        mask = np.full((height, width), 255, dtype=np.uint8)

        num_shapes = random.randint(4, 8)
        max_size = min(height, width) // 8

        for _ in range(num_shapes):
            shape_type = random.choice(["circle", "ellipse", "rectangle"])
            if shape_type == "circle":
                radius = random.randint(max_size // 4, max_size)
                center = (random.randint(radius, width - radius), random.randint(radius, height - radius))
                cv2.circle(mask, center, radius, 0, thickness=-1)

            elif shape_type == "ellipse":
                center = (random.randint(max_size, width - max_size), random.randint(max_size, height - max_size))
                axes = (random.randint(max_size // 4, max_size), random.randint(max_size // 4, max_size))
                angle = random.randint(0, 360)
                cv2.ellipse(mask, center, axes, angle, 0, 360, 0, thickness=-1)

            elif shape_type == "rectangle":
                top_left = (random.randint(0, width - max_size), random.randint(0, height - max_size))
                bottom_right = (top_left[0] + random.randint(max_size // 4, max_size),
                                top_left[1] + random.randint(max_size // 4, max_size))
                cv2.rectangle(mask, top_left, bottom_right, 0, thickness=-1)

        img[mask == 0] = [255, 255, 255]

        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        blurred_mask = cv2.GaussianBlur(dilated_mask, (9, 9), sigmaX=0)
        return Image.fromarray(img), Image.fromarray(blurred_mask)

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

        if self.transform:
            image = self.transform(image)

        damaged_image, mask = self.apply_damage(image)

        if self.to_input:
            image = self.to_input(image)
            damaged_image = self.to_input(damaged_image)
            mask = self.to_input(mask)

        damaged_image = torch.cat([damaged_image, mask], dim=0)

        return damaged_image, image

    def __len__(self):
        return len(self.df)