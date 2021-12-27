import streamlit as st
from drive_connect import load_cpu_model
from PIL import Image
import numpy as np
from api import *

st.title('Denoising(Dehazing) Transformer')
st.write('konkuk graduate project')

with st.spinner("Loading Model.."):
    model = load_cpu_model()

raw_img = st.file_uploader("Upload Haze Image")

if raw_img:
    image = Image.open(raw_img).convert("RGB")
    inference_img = img_transform(image)

inference_run = st.button('Dehaze')

if inference_run:
    output_img = inference(model, inference_img)

    del(model)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Hazy Image")
        st.image(raw_img)

    with col2:
        st.header("Clean Image")
        st.image(output_img)
        st.markdown(get_image_download_link(output_img,raw_img.name,'Download '+raw_img.name), unsafe_allow_html=True)


with st.expander("Sample Result"):
    col1, col2 = st.columns(2)

    with col1:
        st.header("Hazy Image")
        st.image("hazy.jpg")

    with col2:
        st.header("Clean Image")
        st.image("target.jpg")
    