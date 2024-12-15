import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import io
import random
import logging


df = pd.read_parquet('local_wikiart.parquet', columns=['title', 'artist', 'date', 'genre', 'style', 'image'])

def decode_image(image_dict):
    if 'bytes' in image_dict:
        img_bytes = image_dict['bytes']
        img = Image.open(io.BytesIO(img_bytes))
        return img
    return None


class WikiArtDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_dict = self.df.loc[idx, 'image']
        image = decode_image(image_dict)

        if image is None:
            return self.__getitem__((idx + 1) % len(self.df))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
