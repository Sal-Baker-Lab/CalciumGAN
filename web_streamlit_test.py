import streamlit as st
from PIL import Image
import os
import glob
import pandas as pd
from st_aggrid import AgGrid
import random
import base64
import shutil
from random import randint

# Streamlit Page Configuration
dirname = os.path.dirname(__file__)
im = Image.open(dirname + "/favicon.ico")
st.set_page_config(page_title="Calcium GAN", page_icon=im, layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center; color: black;'>Calcium GAN</h1>", unsafe_allow_html=True)


run_container = st.sidebar.container()
run_container_form = st.sidebar.container()
previous_run_container = st.sidebar.container()
export_container = st.sidebar.container()

main_container = st.container()
quant_csv_expander = main_container.expander(label='Click to expand and view Quant result')
col1, col2, col3, col4, col5, col6 = main_container.columns(6)

def genereate_widget_key():
    st.session_state.file_uploader_widget = str(randint(1000, 100000000))
    print(st.session_state.file_uploader_widget)

if 'file_uploader_widget' not in st.session_state:
    genereate_widget_key()

def params():
    return "S_{}_T_{}_C_{}".format(stride_selector, threshold_selector, connectivity_selector)

def run_id():
    return str(random.randrange(0, 1000000, 2))


def refresh_runs_dir():
    runs = filter( os.path.isfile ,glob.glob(dirname + '/runs/*original*'))
    runs = sorted(runs, key = os.path.getmtime, reverse=True)
    dir = tuple(map(lambda  x:  os.path.basename(x), runs))
    st.session_state.runs = dir

def process(input_image, original_image_name, weight_name='000090', stride=16, crop_size=64, thresh=50, connectivity=8):
    # predict.process(input_image, run_directory, weight_name, stride, crop_size, thresh, connectivity)
    predicted_image_name = run_dir  + input_image_name.replace('_original_', '_prediction_')
    threshold_image_name = run_dir  + input_image_name.replace('_original_', '_threshold_')
    input_image.save(predicted_image_name)
    input_image.save(threshold_image_name)
    # print(predicted_image_name)

if 'runs' not in st.session_state:
    refresh_runs_dir()


# Export Container
def create_download_zip(zip_directory, zip_destination, filename):
    if os.path.exists(zip_destination + '/' + filename + '.zip'):
        os.remove(zip_destination + '/' + filename + '.zip')
    shutil.make_archive(zip_destination + '/' + filename, 'zip', zip_directory)
    with open(zip_destination + '/' + filename + '.zip', 'rb') as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download=\'{filename}.zip\'>download file </a>'
        export_container.markdown(href, unsafe_allow_html=True)

download_runs = export_container.button(label='Zip and Export All Runs')
if download_runs:
    run_dir = dirname + "/runs/"
    create_download_zip(run_dir, dirname + '/tmp', 'GanCalcium')



# Run Container

input_image_buffer = run_container.file_uploader("Upload an image", accept_multiple_files=False, type=["jpg", "jpeg"], key=st.session_state.file_uploader_widget)
threshold_selector = run_container.slider('Threshold' , min_value=3 , max_value=254 , value=6 , step=1)
connectivity_selector = run_container.slider('Connectivity' , min_value=4 , max_value=8 , value=4 , step=4)

if input_image_buffer is not None:
    input_image = Image.open(input_image_buffer)
    st.session_state.runs = set()
    col1.header("Input Image")
    col1.image(input_image, use_column_width=True)
    w, h = input_image.size
    stride_selector = run_container.slider('Stride' , min_value=0 , max_value= w-64 , value=3 , step=1)
    submit_button = run_container.button(label='Run Prediction')

    if submit_button:
        run_dir = dirname + "/runs/"
        original_image_name = input_image_buffer.name
        run_id = run_id()
        params = params()
        st.session_state.file_uploader_widget = str(randint(1000, 100000000))
        print(st.session_state.file_uploader_widget)
        input_image_name = run_id + '_original_' + params + original_image_name
        input_image.save(run_dir  + input_image_name)
        refresh_runs_dir()
        process(input_image, input_image_name)

# Previous Runs Selection111
option = previous_run_container.selectbox('Select Run',  options = st.session_state.runs)
if option is not None:
    run_dir = dirname + "/runs/"
    input_image_filename = os.path.join(run_dir, option)
    pred_image_filename = os.path.join(run_dir, option.replace('_original_', '_prediction_'))
    thresh_image_filename = os.path.join(run_dir, option.replace('_original_', '_threshold_'))
    quant_filename = os.path.join(run_dir, option.replace('_original_', '_quant_'))
    quant_filename = quant_filename.replace('.jpg', '.csv')


    if os.path.isfile(input_image_filename):
        input_image = Image.open(input_image_filename)
        col1.header("Run Input Image")
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
