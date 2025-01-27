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

# artworks = [wikiart_dataset[i] for i in range(8)]
# # selected_artwork = display_clickable_artworks(artworks)
# selected_artwork = display_image_select(artworks)

if "current_batch" not in st.session_state:
    st.session_state.current_batch = 0

batch_size = 8
total_batches = (len(dataset) + batch_size - 1) // batch_size
start_idx = st.session_state.current_batch * batch_size
end_idx = min(start_idx + batch_size, len(dataset))

artworks = [wikiart_dataset[i] for i in range(start_idx, end_idx)]

selected_artwork = display_clickable_artworks(artworks)

col1, col2, col3 = st.columns([1, 4, 1])
with col1:
        if st.button("⬅️ Previous") and st.session_state.current_batch > 0:
            st.session_state.current_batch -= 1
            st.rerun()

with col3:
        if st.button("Next ➡️") and st.session_state.current_batch < total_batches - 1:
            st.session_state.current_batch += 1
            st.rerun()

def show_artwork_details(artwork : Artwork):
    with st.expander('Artwork Details', expanded=True):
        f'## {artwork.title}'
        col1, col2, col3 = st.columns(3)
        with col1:
            f'*{artwork.artist}, {artwork.date if artwork.date != "None" else "Unknown"}*'
        with col2:
            f'Style: :primary-background[{artwork.style.capitalize()}]'
        with col3:
            f'Genre: :primary-background[{artwork.genre.capitalize()}]'

# with st.sidebar.container(border=True):
#     col1, col2 = st.columns(2, vertical_alignment='center')
#     with col1:
#         st.number_input("Choose image", min_value=0, max_value=len(wikiart_dataset),
#               value=st.session_state.rn, key='chosen_artwork', on_change=choose_artwork)
#     with col2:
#         st.button("🎨 Show random artwork", on_click=roll_artwork)

if selected_artwork:
    with st.container(border=True):
        '## :primary[What do you want me to do?]'
        is_inpainted = st.checkbox('🖌️ Inpaint')
        is_super = st.checkbox('🪄 Superresolution')

    reconstruction_module = ReconstructionModule()
    # artwork = get_artwork(wikiart_dataset)
    artwork = selected_artwork
    artwork = reconstruction_module.pipeline(artwork, is_inpainted=is_inpainted, is_super=is_super)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')
    with col2:
        if is_inpainted or is_super:
            img1, img2 = align_dimensions(artwork.image, artwork.result)
            image_comparison(
                img1=img1,
                label1="Damaged artwork",
                img2=img2,
                label2="Reconstructed artwork",
                width=600
            )
        else:
            st.image(artwork.image, width=600)
    with col3:
        st.write(' ')

    show_artwork_details(artwork)

# in another tab show other images from category
# choosing images from gallery