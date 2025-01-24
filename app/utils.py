import random
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import streamlit as st
from datasets import load_dataset


@st.cache_resource
def load_wikiart():
    ds = load_dataset("Artificio/WikiArt_Full", split="train")
    return ds[:5]

@st.cache_data
def apply_damage_to_dataset(_dataset):
    return [
        apply_damage(image.resize((224, 224))) for image in _dataset["image"]
    ]

def get_artwork(data):
    return data[st.session_state.rn]

def apply_damage(image):
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
    return Image.fromarray(img)

def roll_artwork(data_len=10):
    st.session_state["rn"] = random.randint(0, data_len)

def choose_artwork():
    st.session_state["rn"] = st.session_state['chosen_artwork']
