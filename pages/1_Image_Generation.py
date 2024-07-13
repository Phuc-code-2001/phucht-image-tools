import streamlit as st

from utils.image_processing import save_PIL_image
from models.diffusion_pipelines import pipeline, device


st.write("# Image Generation with Diffusers")
st.subheader(f"Device: {device}")

st.write('---')
layout = st.columns(2)
text = layout[0].text_area("Enter text to generate image", "A cat sit on a table in a room near a window.")

sub_layout = layout[1].columns(2)
img_width = sub_layout[0].number_input("Image width", min_value=64, max_value=1024, value=256, step=8, key="img_width")
img_height = sub_layout[1].number_input("Image height", min_value=64, max_value=1024, value=256, step=8, key="img_height")

advanced_options = st.expander("Advanced options", expanded=False)
with advanced_options:
    sub_layout = advanced_options.columns(2)
    num_inference_steps = sub_layout[0].number_input("Number of inference steps", min_value=1, max_value=100, value=50, step=1, key="num_inference_steps")
    num_images_per_prompt = sub_layout[1].number_input("Number of images per prompt", min_value=1, max_value=10, value=1, step=1, key="num_images_per_prompt")

st.write('---')

if st.button("Generate", key="generate_button"):

    images = pipeline(
        prompt=text,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        width=img_width,
        height=img_height,
    ).images
    st.image(images[0], use_column_width=512, caption=text)
    save_PIL_image(images[0], "generated_image.jpg")
