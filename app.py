import streamlit as st
from streamlit_image_comparison import image_comparison
from app.models import ReconstructionModule
from app.utils import *

st.set_page_config(
    page_title="Artworks Reconstructions",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

'# :primary[Artworks Reconstructions]'
'Use Neural Networks to restore paintings to their former glory'

with st.spinner('Loading dataset'):
    dataset = load_wikiart()
    wikiart_dataset = WikiArtDataset(dataset)

if "rn" not in st.session_state:
    st.session_state["rn"] = 0

def show_artwork_details(artwork : Artwork):
    with st.expander('Artwork Details', expanded=True):
        f'## {artwork.title}'
        f'*{artwork.artist}, {artwork.date if artwork.date != "None" else "Unknown"}*'
        col1, col2 = st.columns(2)
        with col1:
            f'Style: :primary-background[{artwork.style.capitalize()}]'
        with col2:
            f'Genre: :primary-background[{artwork.genre.capitalize()}]'

with st.sidebar.container(border=True):
    col1, col2 = st.columns(2, vertical_alignment='center')
    with col1:
        st.number_input("Choose image", min_value=0, max_value=len(wikiart_dataset),
              value=st.session_state.rn, key='chosen_artwork', on_change=choose_artwork)
    with col2:
        st.button("🎨 Show random artwork", on_click=roll_artwork)

is_inpainted = st.checkbox('🖌️ Inpaint')
is_super = st.checkbox('🪄 Superresolution')

reconstruction_module = ReconstructionModule()
artwork = get_artwork(wikiart_dataset)
artwork = reconstruction_module.pipeline(artwork, is_inpainted=is_inpainted, is_super=is_super)

if is_inpainted or is_super:
    img1, img2 = align_dimensions(artwork.image, artwork.result)
    image_comparison(
        img1=img1,
        label1="Damaged artwork",
        img2=img2,
        label2="Reconstructed artwork",
    )
else:
    st.image(artwork.image)

show_artwork_details(artwork)

# in another tab show other images from category
# choosing images from gallery