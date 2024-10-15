import streamlit as st
import datasets as ds
from random import randint

@st.cache_resource
def load_wikiart():
    dataset = ds.load_dataset("parquet", data_files={'train': "local_wikiart.parquet"})
    dataset = dataset.cast_column("image", ds.Image(mode="RGB"))
    return dataset['train']


with st.spinner('Loading dataset'):
    df = load_wikiart()

def show_artwork(artwork):
    st.title(artwork["title"])
    f'Artist: *{artwork["artist"]}*'
    col1, col2 = st.columns(2)
    with col1:
        f'Style: {artwork["style"]}'
        f'Genre: {artwork["genre"]}'
    with col2:
        artwork["date"]
    artwork['image']

x = 0
if st.button('Show random artwork', icon="🎨"):
    x = randint(0, len(df))
x
show_artwork(df[x])
