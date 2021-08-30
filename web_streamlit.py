import streamlit as st
import numpy as np
import cv2
from PIL import Image

weights = {
    'global weights': 'global_model_000090.h5',
    'local weights': 'local_model_000090.h5'
}

st.set_page_config(page_title="Calcium GAN", page_icon="🧊", layout="wide",initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center; color: blue;'>Calcium GAN</h1>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
# Config
# st.title("Calcium GAN")
weights_selector = st.sidebar.selectbox("Weight File", list(weights.keys()))
stride_selector = st.sidebar.slider('slider' , min_value=0 , max_value=10 , value=3 , step=1)
crop_selector = st.sidebar.slider('crop' , min_value=0 , max_value=200 , value=64 , step=1)
img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg"])


# file uploader
# uploaded_file = st.file_uploader("Choose a image file", type="jpg")
if img_file_buffer is not None:
    print('test')
    image = Image.open(img_file_buffer)
    col1.header("Original")
    col1.image(image, use_column_width=True)

# print(image)
# img_array = np.array(image)
#
# if image is not None:
#     st.image(
#         image,
#         caption=f"You amazing image has shape {img_array.shape[0:2]}",
#         use_column_width=True,
#     )
