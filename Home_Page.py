import streamlit as st
from PIL import Image
import configure
import os


st.write("# Image Generation with Diffusers")
sample_images = os.listdir("./images/")
sample_images = [os.path.join("./images/", filename) for filename in sample_images]
sample_images = [Image.open(image).resize((300, 300)) for image in sample_images]

st.write("---")

n_cols = 3
layout = st.columns(n_cols)
for i, image in enumerate(sample_images):
    with layout[i % n_cols]:
        st.image(image, use_column_width=True, caption=f"Image {i+1}")
