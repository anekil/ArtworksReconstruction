import numpy as np
import torch
import random
from torchvision import transforms

from PIL import Image

from UNN.dataloading.WikiArtDataset import WikiArtDataset

def apply_damage(image):
    img = np.array(image)
    height, width, _ = img.shape
    mask_size = int(min(height, width) // 4)
    top_left_x = random.randint(0, width - mask_size)
    top_left_y = random.randint(0, height - mask_size)
    img[top_left_y:top_left_y + mask_size, top_left_x:top_left_x + mask_size] = [255, 255, 255]
    return Image.fromarray(img)

class InpaintingDataset(WikiArtDataset):
    def __init__(self, df, transform=None, mask_size=(50, 50)):
        super().__init__(df, transform)
        self.mask_size = mask_size

    def __getitem__(self, idx):
        image = super().__getitem__(idx)

        damaged_image = apply_damage(image)
        resize_transform = transforms.Resize((224, 224))
        image_resized = resize_transform(image)
        if self.transform:
            damaged_image = self.transform(damaged_image)
            original_image = self.transform(image_resized)
        else:
            to_tensor = transforms.ToTensor()
            damaged_image = to_tensor(damaged_image)
            original_image = to_tensor(image_resized)

        return damaged_image, original_image

    def apply_random_mask(self, image):
        _, H, W = image.size()
        mask = torch.ones_like(image)

        mask_height = random.randint(H // 8, H // 4)
        mask_width = random.randint(W // 8, W // 4)
        top = random.randint(0, H - mask_height)
        left = random.randint(0, W - mask_width)

        mask[:, top:top + mask_height, left:left + mask_width] = 0
        masked_image = image * mask

        return masked_image
