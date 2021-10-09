import streamlit as st
from PIL import Image
import os
import glob
import pandas as pd
from st_aggrid import AgGrid
import streamlit.components.v1 as components
import random
import base64
import shutil
from random import randint
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit Page Configuration
dirname = os.path.dirname(__file__)
im = Image.open(dirname + "/favicon.ico")
st.set_page_config(page_title="Calcium GAN", page_icon=im, layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align: center; color: black;'>Calcium GAN</h1>",
            unsafe_allow_html=True)

run_container = st.sidebar.container()
run_container_form = st.sidebar.container()
calibration_container = st.sidebar.container()
st.sidebar.markdown("""---""")
previous_run_container = st.sidebar.container()
st.sidebar.markdown("""---""")
st.sidebar.markdown("Export All Runs")
export_container = st.sidebar.container()

main_container = st.container()
quant_csv_expander = main_container.expander(
    label='Click to expand and view Quant result')
calibrated_quant_csv_expander = main_container.expander(
    label='Click to expand and view Calibrated Quant result')
plots_quant_csv_expander = main_container.expander(
    label='Click to view plots')
plot_col1, plot_col2, plot_col3, plot_col4 = plots_quant_csv_expander.columns(4)

col1, col2, col3, col4, col5, col6 = main_container.columns(6)


def display_plot(col, plot_type):
    with plots_quant_csv_expander:
        plot_path = run_dir + "/" + option + plot_type
        if os.path.isfile(plot_path):
            plot_image = Image.open(plot_path)
            col.image(plot_image, use_column_width=True)


def display_predictions(col, original_image_path, label, image_type):
    image_path = original_image_path.replace("_original_", image_type)
    if os.path.isfile(image_path):
        image = Image.open(image_path)
        col.header(label)
        col.image(resize_displayed_image(image), use_column_width=False)


def resize_displayed_image(image, fixed_height=500):
    height_percent = (fixed_height / float(image.size[1]))
    width_size = int((float(image.size[0]) * float(height_percent)))
    image1 = image.resize((width_size, fixed_height), Image.NEAREST)
    return image1


def interval(df):
    df['Interval'] = df.apply(
        lambda x: abs(x['Top'] - (x.shift(1)['Top'] + x.shift(1)['Height'])),
        axis=1)
    return df


def genereate_widget_key():
    st.session_state.file_uploader_widget = str(randint(1000, 100000000))
    print(st.session_state.file_uploader_widget)


if 'file_uploader_widget' not in st.session_state:
    genereate_widget_key()


# def generate_plots(calibrated_quant_csv):
#     with plots_quant_csv_expander:
#         if os.path.isfile(calibrated_quant_filename):
#             dataframe = pd.read_csv(calibrated_quant_filename)
#             dataframe = dataframe.assign(category='')
#             dataframe = interval(dataframe)
#             #
#             plt.margins(x=0)
#             plt.yticks(fontsize=16)
#             sns.set(font_scale = 1.5)
#             #
#
#             fig1, ax1 = plt.subplots(squeeze=True)
#             sns.barplot(x='category', y='Frequency', data=dataframe,
#                         dodge=True, palette='viridis', ax = ax1)
#
#             # sns.despine(top=True, right=True, left=False, bottom=False)
#             ax1.set_xlabel('')
#             # ax1.set_ylabel('Frequency No. of ' + r'$Ca^2+ Events$' +'\n (per STMap)',  fontsize = 18)
#             ax1.set_facecolor('xkcd:white')
#
#             plot_col1.pyplot(fig1)
#             #
#             fig2, ax2 = plt.subplots(squeeze=True)
#             sns.swarmplot(x='category', y='Area', data=dataframe,
#                           dodge=True, palette='viridis', ax = ax2)
#             # sns.despine(top=True, right=True, left=False, bottom=False)
#             ax2.set_xlabel('')
#             # ax2.set_ylabel(r'Area ($\mu$m*s)',  fontsize = 20)
#             plot_col2.pyplot(fig2)
#             #     #
#             #
#             fig3, ax3 = plt.subplots(squeeze=True)
#
#             sns.swarmplot(x='category', y='Height', data=dataframe,
#                           dodge=True, palette='viridis', ax = ax3)
#             # sns.despine(top=True, right=True, left=False, bottom=False)
#             ax3.set_xlabel('')
#             ax3.set_ylabel(r'Duration - Time ($\mu$s)', fontsize = 20)
#             plot_col3.pyplot(fig3)
# #
# #
#
#             fig4, ax4 = plt.subplots(squeeze=True)
#             sns.swarmplot(x='category', y='Interval', data=dataframe,
#                           dodge=True, palette='viridis', ax = ax4)
#             # sns.despine(top=True, right=True, left=False, bottom=False)
#
#             ax4.set_xlabel('')
#             ax4.set_ylabel('Spatial spread - Distance \n' + r'$(mu*s)$', fontsize = 20)
#             ax4.set_xmargin(0)
#             # ax4.margins(x=0, tight=None)
#
#             plot_col4.pyplot(fig4)
#         else:
#             dataframe = None


def params():
    return "S_{}_T_{}_C_{}".format(stride_selector, threshold_selector,
                                   connectivity_selector)


def run_id():
    return str(random.randrange(0, 1000000, 2))


def refresh_runs_dir():
    runs = filter(os.path.isdir, glob.glob(dirname + '/runs/*'))
    runs = sorted(runs, key=os.path.getmtime, reverse=True)
    dir = tuple(map(lambda x: os.path.basename(x), runs))
    st.session_state.runs = dir


def process(run_dir, weight_name='000090',
    stride=16, crop_size=64, thresh=50, connectivity=8, alpha=0.7,
    height_calibration=1,
    width_calibration=1):
    # predict.process(input_image, run_directory, weight_name, stride, crop_size, thresh, connectivity)
    input_images = list(
        filter(os.path.isfile, glob.glob(run_dir + f"/*_original_*")))

    for image_path in input_images:
        image = Image.open(image_path)
        predicted_image_name =  image_path.replace('_original_',
                                                            '_prediction_')

        threshold_image_name = image_path.replace('_original_',
                                                            '_threshold_')

        overlay_image_name = image_path.replace('_original_',
                                                          '_overlay_')
        image.save(predicted_image_name)
        image.save(threshold_image_name)
        image.save(overlay_image_name)


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

input_image_buffer = run_container.file_uploader("Upload an image",
                                                 accept_multiple_files=True,
                                                 type=["jpg", "jpeg"],
                                                 key=st.session_state.file_uploader_widget)
threshold_selector = run_container.slider('Threshold', min_value=3,
                                          max_value=254, value=6, step=1)
connectivity_selector = run_container.slider('Connectivity', min_value=4,
                                             max_value=8, value=4, step=4)
height_calibration_selector = run_container.slider('Height Calibration px',
                                                   min_value=1,
                                                   max_value=10, value=1,
                                                   step=1)
width_calibration_selector = run_container.slider('Width Calibration px',
                                                  min_value=1,
                                                  max_value=10, value=1, step=1)

if input_image_buffer is not None and len(input_image_buffer) > 0:
    first_input_image = Image.open(input_image_buffer[0])
    st.session_state.runs = set()
    # col1.header("Selected Image")
    # col1.image(input_image, use_column_width=True)
    w, h = first_input_image.size
    stride_selector = run_container.slider('Stride', min_value=0,
                                           max_value=w - 64, value=3, step=1)
    submit_button = run_container.button(label='Run Prediction')

    if submit_button:
        run_dir = dirname + "/runs/"
        # original_image_name = input_image_buffer.name

        # generate run_id
        run_id = run_id()
        params = params()
        st.session_state.file_uploader_widget = str(randint(1000, 100000000))

        # rename save input images
        os.mkdir(run_dir + run_id)
        for image_path in input_image_buffer:
            new_image_name = run_id + '_original_' + params + image_path.name
            image = Image.open(image_path)
            image.save(run_dir + run_id + "/" + new_image_name)

        refresh_runs_dir()
        process(run_dir + run_id + "/")

# Previous Runs Selection111
option = previous_run_container.selectbox('Select Run',
                                          options=st.session_state.runs)
if option is not None:
    run_dir = dirname + "/runs/" + option
    input_images = list(
        filter(os.path.isfile, glob.glob(run_dir + f"/*_original_*")))

    for input_image in input_images:
        display_predictions(col1, input_image, 'Input Image', "_original_")
        display_predictions(col2, input_image, 'Predicted Image',
                            "_prediction_")
        display_predictions(col3, input_image, 'Threshold Image', "_threshold_")
        display_predictions(col4, input_image, 'Overlay Image', "_overlay_")

        quant_filename = input_image.replace('_original_', '_quant_')
        quant_filename = quant_filename.replace('.jpg', '.csv')

        calibrated_quant_filename = input_image.replace('_original_',
                                                        '_calibrated_quant_')
        calibrated_quant_filename = calibrated_quant_filename.replace('.jpg',
                                                                      '.csv')

    with quant_csv_expander:
        if os.path.isfile(quant_filename):
            dataframe = pd.read_csv(quant_filename)
            AgGrid(dataframe, height=500, fit_columns_on_grid_load=True)
        else:
            dataframe = None
    with calibrated_quant_csv_expander:
        if os.path.isfile(calibrated_quant_filename):
            dataframe = pd.read_csv(calibrated_quant_filename)
            AgGrid(dataframe, height=500, fit_columns_on_grid_load=True)
        else:
            dataframe = None

    display_plot(plot_col1, "_frequency.png")
    display_plot(plot_col2, "_area.png")
    display_plot(plot_col3, "_distance.png")
    display_plot(plot_col4, "_duration.png")
