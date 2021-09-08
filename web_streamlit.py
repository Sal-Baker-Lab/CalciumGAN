import streamlit as st
from PIL import Image
import os
import glob
import datetime
import predict
import asyncio
import pandas as pd
from st_aggrid import AgGrid
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True

dirname = os.path.dirname(__file__)


def refresh_runs_dir():
    runs = filter( os.path.isdir,glob.glob(dirname + '/runs/*'))
    runs = sorted(runs, key = os.path.getmtime, reverse=True)
    st.session_state.runs = tuple(map(lambda  x:  os.path.basename(x), runs))

if 'runs' not in st.session_state:
    refresh_runs_dir()



im = Image.open(dirname + "/favicon.ico")
st.set_page_config(page_title="Calcium GAN", page_icon=im, layout="wide",initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center; color: black;'>Calcium GAN</h1>", unsafe_allow_html=True)

quant_csv_expander = st.expander(label='Quant CSV')

input_image_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg"])
option = st.sidebar.selectbox('Select Run',  st.session_state.runs)

col1, col2, col3= st.columns(3)
# Config
# st.title("Calcium GAN")
threshold_selector = st.sidebar.slider('Threshold' , min_value=3 , max_value=254 , value=6 , step=1)
connectivity = st.sidebar.slider('Connectivity' , min_value=4 , max_value=8 , value=4 , step=4)


def process(input_image, run_directory, weight_name='000090', stride=16, crop_size=64, thresh=50, connectivity=8):
    predict.process(input_image, run_directory, weight_name, stride, crop_size, thresh, connectivity)


def refresh_runs_dir():
    runs = filter( os.path.isdir,glob.glob(dirname + '/runs/*'))
    runs = sorted(runs, key = os.path.getmtime, reverse=True)
    st.session_state.runs = tuple(map(lambda  x:  os.path.basename(x), runs))

if 'runs' not in st.session_state:
    refresh_runs_dir()


if option is not None:
    run_dir = dirname + "/runs/" + option
    input_image_filename = os.path.join(run_dir, 'input_image.jpg')
    pred_image_filename = os.path.join(run_dir, 'pred_image.jpg')
    thresh_image_filename = os.path.join(run_dir, 'thresh_image.jpg')
    quant_filename = os.path.join(run_dir, 'quant_csv.csv')

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

    with quant_csv_expander:
        if os.path.isfile(quant_filename):
            dataframe = pd.read_csv(quant_filename)
        else:
            dataframe = None
        AgGrid(dataframe, height=500, fit_columns_on_grid_load=True)


def form_callback():
    refresh_runs_dir()

with st.sidebar.form(key='run_form'):
    if input_image_buffer is not None:
        input_image = Image.open(input_image_buffer)
        col1.header("Input Image")
        col1.image(input_image, use_column_width=True)
        w, h = input_image.size
        stride_selector = st.sidebar.slider('Stride' , min_value=0 , max_value= w-64 , value=3 , step=1)

        # user_input = st.sidebar.text_input("Run label", "run_")
        # submit_button = st.form_submit_button(label='Submit', on_click=form_callback(input_image))
        submit_button = st.form_submit_button(label='Submit', on_click=form_callback)

        if submit_button:
            run_dir = dirname + "/runs/"
            run_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            run_directory = os.path.join(run_dir, str(run_timestamp))
            os.mkdir(run_directory)
            input_image.save(run_directory + '/input_image.jpg')
            refresh_runs_dir()
            process(input_image, run_timestamp)
