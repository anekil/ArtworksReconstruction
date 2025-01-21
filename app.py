import streamlit as st
import datasets as ds
from random import randint
import numpy as np
from PIL import Image
from streamlit_image_comparison import image_comparison

'# :rainbow[Artworks Reconstructions]'

@st.cache_resource
def load_wikiart():
    dataset = ds.load_dataset("parquet", data_files={'train': "local_wikiart.parquet"})
    dataset = dataset.cast_column("image", ds.Image(mode="RGB"))
    return dataset['train']


with st.spinner('Loading dataset'):
    df = load_wikiart()

if "rn" not in st.session_state:
    st.session_state["rn"] = randint(0,len(df))

def roll_artwork(number=None):
    st.session_state["rn"] = randint(0,len(df))

def choose_artwork():
    st.session_state["rn"] = st.session_state['chosen_artwork']

def apply_damage(image):
    img = np.array(image)
    height, width, _ = img.shape
    mask_size = int(min(height, width) // 4)
    top_left_x = randint(0, width - mask_size)
    top_left_y = randint(0, height - mask_size)
    img[top_left_y:top_left_y + mask_size, top_left_x:top_left_x + mask_size] = [255, 255, 255]
    return Image.fromarray(img)

def show_artwork_details(artwork):
    with st.expander('Artwork Details', expanded=True):
        f'## {artwork["title"]}'
        f'*{artwork["artist"]}, {artwork["date"] if artwork["date"] != "None" else "Unknown"}*'
        col1, col2 = st.columns(2)
        with col1:
            f'Style: :rainbow-background[{artwork["style"].capitalize()}]'
        with col2:
            f'Genre: :rainbow-background[{artwork["genre"].capitalize()}]'


with st.container(border=True):
    col1, col2 = st.columns(2, vertical_alignment='center')
    with col1:
        st.slider("Choose image", min_value=0, max_value=len(df),
              value=st.session_state.rn, key='chosen_artwork', on_change=choose_artwork)
    with col2:
        st.button("🎨Show random artwork", on_click=roll_artwork)
    is_damaged = st.checkbox('💥 Show damaged artwork')
    if is_damaged:
        is_inpainted = st.checkbox('🖌️ Inpaint')

artwork = df[st.session_state.rn]
image = artwork['image']

if is_damaged:
    show_artwork_details(artwork)
    damaged = apply_damage(image)
    if is_inpainted:
        image_comparison(
            img1=damaged,
            label1="Damaged artwork",
            img2=image,
            label2="Reconstructed artwork",
            width=700
        )
    else:
        st.image(damaged, width=700)
else:
    show_artwork_details(artwork)
    st.image(image, width=700)
