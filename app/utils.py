import random
import numpy as np
from PIL import Image
import cv2
import streamlit as st
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import v2
import io
import base64
from st_clickable_images import clickable_images


class Artwork:
    def __init__(self, image, mask, title: str, artist: str, date: str, style: str, genre: str):
        self.image = image
        self.mask = mask
        self.title = title
        self.artist = artist
        self.date = date
        self.style = style
        self.genre = genre
        self.result = None


@st.cache_resource
def load_wikiart():
    ds = load_dataset("Artificio/WikiArt_Full", split="train")
    return ds

def get_artwork(data):
    return data[st.session_state["rn"]]

class WikiArtDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = v2.Compose([
            v2.Resize((256, 256)),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            data = self.dataset[idx]
            return self._process_data(data, idx)
        elif isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self._process_data(self.dataset[i], i) for i in indices]
        elif isinstance(idx, list):
            return [self._process_data(self.dataset[i], i) for i in idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def _process_data(self, data, idx):
        image = self.transform(data["image"])
        image, mask = self.apply_damage(image, idx)

        return Artwork(
            image=image,
            mask=mask,
            title=data.get("title"),
            artist=data.get("artist"),
            date=data.get("date"),
            style=data.get("style"),
            genre=data.get("genre"),
        )

    def get_images(self, idx):
        if isinstance(idx, int):
            return self.dataset[idx].image
        elif isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return [self.dataset[i].image for i in indices]
        elif isinstance(idx, list):
            return [self.dataset[i].image for i in idx]

    def apply_damage(self, image, idx):
        seed = hash(str(idx)) % (2**32)
        random.seed(seed)

        img = np.array(image, dtype=np.uint8)
        height, width, _ = img.shape
        mask = np.ones((height, width), dtype=np.uint8) * 255

        max_area = int(0.16 * height * width)
        current_area = 0
        num_shapes = random.randint(6, 12)
        max_size = min(height, width) // 8

        while current_area < max_area and num_shapes > 0:
            shape_type = random.choice(["circle", "ellipse", "rectangle"])
            temp_mask = np.ones_like(mask, dtype=np.uint8) * 255
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
        return Image.fromarray(img), Image.fromarray(mask)

def roll_artwork(data_len=10):
    st.session_state["rn"] = random.randint(0, data_len)

def choose_artwork():
    st.session_state["rn"] = st.session_state['chosen_artwork']

def align_dimensions(image1, image2):
    image1 = np.array(image1)
    image2 = np.array(image2)

    height, width = max(image1.shape[:2], image2.shape[:2])
    image1 = cv2.resize(image1, (width, height))
    image2 = cv2.resize(image2, (width, height))

    return image1, image2

def pil_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_str}"

def display_clickable_artworks(artworks):
    images = [pil_to_base64(artwork.image) for artwork in artworks]

    clicked_index = clickable_images(
        images,
        titles=[artwork.title for artwork in artworks],
        div_style={"display": "flex", "flex-wrap": "wrap"},
        img_style={"margin": "5px", "height": "200px"},
    )

    if clicked_index != -1:
        return artworks[clicked_index]
    return None
