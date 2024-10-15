import streamlit as st
from streamlit_image_comparison import image_comparison
from streamlit_image_coordinates import streamlit_image_coordinates

value = streamlit_image_coordinates("static/1.jpg")
st.write(value)

st.markdown('# Porównanie obrazów')

image_comparison(
    img1="static/1.jpg",
    img2="static/2.jpg",
    width=700,
    starting_position=50,
    show_labels=True,
    make_responsive=True,
    in_memory=True,
)

from streamlit_image_select import image_select
img = image_select("Wybierz obraz", ["static/1.jpg", "static/2.jpg"] * 4)
st.write(img)