import streamlit as st
from app.models import ReconstructionModule
from app.utils import *

st.set_page_config(
    page_title="Artworks Reconstructions",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

'# :rainbow[Artworks Reconstructions]'
'Use Neural Networks to restore paintings to their former glory'

with st.spinner('Loading dataset'):
    dataset = load_wikiart()
    wikiart_dataset = WikiArtDataset(dataset)

if "rn" not in st.session_state:
    st.session_state["rn"] = 0

reconstruction_module = ReconstructionModule()
artwork = get_artwork(wikiart_dataset)
st.image(artwork.image)

inpainted = reconstruction_module.inpainting(artwork)
st.image(inpainted.image)

super_artwork = reconstruction_module.resolution(inpainted)
st.image(super_artwork.image)

# def show_artwork_details():
#     artwork = dataset[st.session_state.rn]
#     with st.expander('Artwork Details', expanded=True):
#         f'## {artwork["title"]}'
#         f'*{artwork["artist"]}, {artwork["date"] if artwork["date"] != "None" else "Unknown"}*'
#         col1, col2 = st.columns(2)
#         with col1:
#             f'Style: :rainbow-background[{artwork["style"].capitalize()}]'
#         with col2:
#             f'Genre: :rainbow-background[{artwork["genre"].capitalize()}]'

# with st.sidebar.container(border=True):
#     col1, col2 = st.columns(2, vertical_alignment='center')
#     with col1:
#         st.number_input("Choose image", min_value=0, max_value=len(damaged_images),
#               value=st.session_state.rn, key='chosen_artwork', on_change=choose_artwork)
#     with col2:
#         st.button("🎨 Show random artwork", on_click=roll_artwork)
#
# artwork = get_artwork(damaged_images)
#
# is_inpainted = st.checkbox('🖌️ Inpaint')
# is_super = st.checkbox('🪄 Superresolution')
#
# if is_inpainted:
#     image_comparison(
#         img1=artwork,
#         label1="Damaged artwork",
#         img2=artwork,
#         label2="Reconstructed artwork",
#         width=700
#     )
# else:
#     st.image(artwork, width=700)

# show_artwork_details()

# in another tab show other images from category
# choosing images from gallery