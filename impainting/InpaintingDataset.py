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
        super().__init__(df)
        self.df = df
        self.transform = transform
        self.mask_size = mask_size

    def __getitem__(self, idx):
        image = super().__getitem__(idx)
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        damaged_image = apply_damage(image)
        if self.transform:
            original_image = self.transform(image)
            damaged_image = self.transform(damaged_image)
        else:
            original_image = image

        return damaged_image, original_image

    def __len__(self):
        return len(self.df)