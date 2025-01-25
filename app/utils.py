import random
import numpy as np
from PIL import Image
import cv2
import streamlit as st
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import v2


class Artwork:
    def __init__(self, image, mask, cluster_id: int, title: str, artist: str, date: str, style: str, genre: str):
        self.image = image
        self.mask = mask
        self.cluster_id = cluster_id
        self.title = title
        self.artist = artist
        self.date = date
        self.style = style
        self.genre = genre


@st.cache_resource
def load_wikiart():
    ds = load_dataset("Artificio/WikiArt_Full", split="train")
    return ds

def get_artwork(data):
    return data[st.session_state.rn]

class WikiArtDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]

        transform = v2.Compose([
            v2.Resize((224, 224)),
        ])

        image = transform(image)
        image, mask = self.apply_damage(image)

        artwork = Artwork(
            image=image,
            mask=mask,
            cluster_id=1,

            title=self.dataset[idx]["title"],
            artist=self.dataset[idx]["artist"],
            date=self.dataset[idx]["date"],
            style=self.dataset[idx]["style"],
            genre=self.dataset[idx]["genre"],
        )

        return artwork

    def apply_damage(self, image):
        img = np.array(image)
        height, width, _ = img.shape
        mask = np.ones((height, width), dtype=np.uint8)

        max_area = int(0.16 * height * width)
        current_area = 0
        num_shapes = random.randint(6, 12)
        max_size = min(height, width) // 8

        while current_area < max_area and num_shapes > 0:
            shape_type = random.choice(["circle", "ellipse", "rectangle"])
            temp_mask = np.ones_like(mask)
            center = (
                random.randint(max_size, width - max_size),
                random.randint(max_size, height - max_size)
            )

            if shape_type == "circle":
                radius = random.randint(max_size // 4, max_size // 2)
                cv2.circle(temp_mask, center, radius, 0, thickness=-1)
            elif shape_type == "ellipse":
                axes = (
                    random.randint(max_size // 6, max_size // 3),
                    random.randint(max_size // 6, max_size // 3)
                )
                angle = random.randint(0, 360)
                cv2.ellipse(temp_mask, center, axes, angle, 0, 360, 0, thickness=-1)
            elif shape_type == "rectangle":
                bottom_right = (
                    center[0] + random.randint(max_size // 6, max_size // 3),
                    center[1] + random.randint(max_size // 6, max_size // 3)
                )
                cv2.rectangle(temp_mask, center, bottom_right, 0, thickness=-1)

            new_area = np.sum(temp_mask == 0)
            if current_area + new_area > max_area:
                break

            mask = cv2.bitwise_and(mask, temp_mask)
            current_area += new_area
            num_shapes -= 1

        img[mask == 0] = [255, 255, 255]
        mask = mask * 255
        return Image.fromarray(img), Image.fromarray(mask)