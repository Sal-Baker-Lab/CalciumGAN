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
import random
tb._SYMBOLIC_SCOPE.value = True

dirname = os.path.dirname(__file__)

def params():
    return "S_{}_T_{}_C_{}".format(stride_selector, threshold_selector, connectivity_selector)

def run_id():
    return str(random.randrange(0, 1000000, 2))


def refresh_runs_dir():
    runs = filter( os.path.isfile ,glob.glob(dirname + '/runs/*original*'))
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
threshold_selector = st.sidebar.slider('Threshold' , min_value=3 , max_value=254 , value=6 , step=1)
connectivity_selector = st.sidebar.slider('Connectivity' , min_value=4 , max_value=8 , value=4 , step=4)

def process(input_image, original_image_name, weight_name='000090', stride=16, crop_size=64, thresh=50, connectivity=8):
    predict.process(input_image, original_image_name, weight_name, stride, crop_size, thresh, connectivity)
    predicted_image_name = run_dir  + input_image_name.replace('_original_', '_prediction_')
    print(predicted_image_name)


if option is not None:
    run_dir = dirname + "/runs/"
    input_image_filename = os.path.join(run_dir, option)
    pred_image_filename = os.path.join(run_dir, option.replace('_original_', '_prediction_'))
    thresh_image_filename = os.path.join(run_dir, option.replace('_original_', '_threshold_'))
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
            AgGrid(dataframe, height=500, fit_columns_on_grid_load=True)
        else:
            dataframe = None


def form_callback():
    print('callback')
    # refresh_runs_dir()


with st.sidebar.form(key='run_form'):
    if input_image_buffer is not None:
        input_image = Image.open(input_image_buffer)
        col1.header("Input Image")
        col1.image(input_image, use_column_width=True)
        w, h = input_image.size
        stride_selector = st.sidebar.slider('Stride' , min_value=0 , max_value= w-64 , value=3 , step=1)
        submit_button = st.form_submit_button(label='Submit', on_click=form_callback)

        if submit_button:
            run_dir = dirname + "/runs/"
            original_image_name = input_image_buffer.name
            run_id = run_id()
            params = params()
            input_image_name = run_id + '_original_' + params + original_image_name
            input_image.save(run_dir  + input_image_name)
            refresh_runs_dir()
            process(input_image, input_image_name)
