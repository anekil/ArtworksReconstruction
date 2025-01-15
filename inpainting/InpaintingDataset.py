import io

import cv2
import numpy as np
import torch
import random

from PIL import Image
from torchvision.transforms import v2


class InpaintingDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def apply_damage(self, image):
        img = np.array(image)
        height, width, _ = img.shape
        mask = np.ones((height, width), dtype=np.uint8) * 255

        num_shapes = random.randint(4, 8)
        max_size = min(height, width) // 8

        for _ in range(num_shapes):
            shape_type = random.choice(["circle", "ellipse", "rectangle"])
            if shape_type == "circle":
                radius = random.randint(max_size // 4, max_size)
                center_x = random.randint(radius, width - radius)
                center_y = random.randint(radius, height - radius)
                cv2.circle(mask, (center_x, center_y), radius, 0, -1)
                cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), -1)

            elif shape_type == "ellipse":
                center_x = random.randint(max_size, width - max_size)
                center_y = random.randint(max_size, height - max_size)
                axis_x = random.randint(max_size // 4, max_size)
                axis_y = random.randint(max_size // 4, max_size)
                angle = random.randint(0, 360)
                cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), angle, 0, 360, 0, -1)
                cv2.ellipse(img, (center_x, center_y), (axis_x, axis_y), angle, 0, 360, (255, 255, 255), -1)

            elif shape_type == "rectangle":
                top_left_x = random.randint(0, width - max_size)
                top_left_y = random.randint(0, height - max_size)
                bottom_right_x = top_left_x + random.randint(max_size // 4, max_size)
                bottom_right_y = top_left_y + random.randint(max_size // 4, max_size)
                cv2.rectangle(mask, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 0, -1)
                cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 255, 255), -1)

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

        if self.transform:
            image = self.transform(image)

        damaged_image, mask = self.apply_damage(image)
        original_image = self.to_tensor(image)
        damaged_image = self.to_tensor(damaged_image)
        mask = self.to_tensor(mask)
        damaged_image = torch.cat([damaged_image, mask], dim=0)

        return damaged_image, original_image

    def __len__(self):
        return len(self.df)