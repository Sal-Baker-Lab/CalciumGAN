import streamlit as st
from PIL import Image
import os
import glob
import datetime

dirname = os.path.dirname(__file__)

st.set_page_config(page_title="Calcium GAN", page_icon="🧊", layout="wide",initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center; color: blue;'>Calcium GAN</h1>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
# Config
st.title("Calcium GAN")
stride_selector = st.sidebar.slider('slider' , min_value=0 , max_value=10 , value=3 , step=1)
crop_selector = st.sidebar.slider('crop' , min_value=0 , max_value=200 , value=64 , step=1)
input_image_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg"])

def predict(run_directory):
    print('test')


def refresh_runs_dir():
    runs = filter( os.path.isdir,glob.glob(dirname + '/runs/*'))
    runs = sorted(runs, key = os.path.getmtime, reverse=True)
    st.session_state.runs = tuple(map(lambda  x:  os.path.basename(x), runs))

if 'runs' not in st.session_state:
    refresh_runs_dir()

# load output of selected run
option = st.sidebar.selectbox('Select Run',  st.session_state.runs)
if option is not None:
    run_dir = dirname + "/runs/" + option
    input_image_filename = os.path.join(run_dir, 'input_image.jpg')
    pred_image_filename = os.path.join(run_dir, 'pred_image.jpg')
    thresh_image_filename = os.path.join(run_dir, 'thresh_image.jpg')

    if os.path.isfile(input_image_filename):
        input_image = Image.open(input_image_filename)
        col1.header("Input Image")
        col1.image(input_image, use_column_width=True)

    if os.path.isfile(pred_image_filename):
        predicted_image = Image.open(pred_image_filename)
        col2.header("Predicted Image")
        col2.image(predicted_image, use_column_width=True)

    if os.path.isfile(thresh_image_filename):
        thresh_image = Image.open(thresh_image_filename)
        col3.header("Threshold Image")
        col3.image(thresh_image, use_column_width=True)


def form_callback(input_image):
    run_dir = dirname + "/runs/"
    run_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    new_directory = os.path.join(run_dir, str(run_timestamp))
    os.mkdir(new_directory)
    input_image.save(new_directory + '/input_image.jpg')
    refresh_runs_dir()
    predict(new_directory)

with st.sidebar.form(key='run_form'):
    if input_image_buffer is not None:
        input_image = Image.open(input_image_buffer)
        col1.header("Input Image")
        col1.image(input_image, use_column_width=True)
        submit_button = st.form_submit_button(label='Submit', on_click=form_callback(input_image))
