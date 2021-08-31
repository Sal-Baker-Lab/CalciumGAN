import streamlit as st
import numpy as np
import cv2
from PIL import Image

def predict_image(image):
    col2.header("Predicted Image")
    col2.image(image, use_column_width=True)
    return image

def post_process_image(image):
    col3.header("Post Process Image")
    col3.image(image, use_column_width=True)



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
input_image_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg"])


if input_image_buffer is not None:
    input_image = Image.open(input_image_buffer)
    col1.header("Original")
    col1.image(input_image, use_column_width=True)
    predicted_image = predict_image(input_image)
    post_process_image(predicted_image)






# print(image)
# img_array = np.array(image)
#
# if image is not None:
#     st.image(
#         image,
#         caption=f"You amazing image has shape {img_array.shape[0:2]}",
#         use_column_width=True,
#     )
