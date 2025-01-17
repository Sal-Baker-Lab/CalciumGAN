import asyncio
import numpy as np
from model import fine_generator, coarse_generator
# from libtiff import TIFF
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import random
import cv2
from functools import partial
import numpy as np
import tensorflow as tf
import keras
import argparse
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Model, load_model
import keras.backend as K
from keras.initializers import RandomNormal
from numpy import load
from sklearn.metrics import confusion_matrix, jaccard_similarity_score, \
    f1_score, roc_auc_score, auc, recall_score, auc, roc_curve
import gc
import glob
import pycm
import stats

import warnings

warnings.filterwarnings('ignore')

# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True
def interval(df):
    df['Interval']=df.apply(lambda x: abs(x['Top'] - (x.shift(1)['Top'] + x.shift(1)['Duration'])), axis=1)
    return df

def remove_image_duplicate_name(df):
    df['Image']=df.apply(lambda x: x['Image'] if x['Frequency'] == 1 else ' ', axis=1)
    return df

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

dirname = os.path.dirname(__file__)


def normalize_pred(img, g_global_model, g_local_model):
    img = np.reshape(img, [1, 64, 64, 1])
    img_coarse = tf.image.resize(img, (32, 32),
                                 method=tf.image.ResizeMethod.LANCZOS3)
    img_coarse = (img_coarse - 127.5) / 127.5
    img_coarse = np.array(img_coarse)

    X_fakeB_coarse, x_global = g_global_model.predict(img_coarse)
    X_fakeB_coarse = (X_fakeB_coarse + 1) / 2.0
    pred_img_coarse = X_fakeB_coarse[:, :, :, 0]

    img = (img - 127.5) / 127.5
    X_fakeB = g_local_model.predict([img, x_global])
    X_fakeB = (X_fakeB + 1) / 2.0
    pred_img = X_fakeB[:, :, :, 0]
    return [np.asarray(pred_img, dtype=np.float32),
            np.asarray(pred_img_coarse, dtype=np.float32)]


def strided_crop(img, img_h, img_w, height, width, g_global_model,
    g_local_model, stride=1):
    full_prob = np.zeros((img_h, img_w), dtype=np.float32)
    full_sum = np.ones((img_h, img_w), dtype=np.float32)

    max_x = int(((img.shape[0] - height) / stride) + 1)
    max_y = int(((img.shape[1] - width) / stride) + 1)
    max_crops = (max_x) * (max_y)
    i = 0
    for h in range(max_x):
        for w in range(max_y):
            crop_img_arr = img[h * stride:(h * stride) + height,
                           w * stride:(w * stride) + width]
            [pred, pred_256] = normalize_pred(crop_img_arr, g_global_model,
                                              g_local_model)
            full_prob[h * stride:(h * stride) + height,
            w * stride:(w * stride) + width] += pred[0]
            full_sum[h * stride:(h * stride) + height,
            w * stride:(w * stride) + width] += 1
            i = i + 1
            # print(i)
    out_img = full_prob / full_sum
    return out_img


def threshold(img, thresh):
    binary_map = (img > thresh).astype(np.uint8)
    binary_map[binary_map == 1] = 255
    return binary_map


def overlay(img, mask, alpha=0.7):
    overlay = np.zeros((mask.shape[0], mask.shape[1], 3))
    overlay[mask == 255] = 255
    overlay[:, :, 1] = 0
    overlay[:, :, 2] = 0
    image = np.zeros((img.shape[0], img.shape[1], 3))
    image[:, :, 0] = img
    image[:, :, 1] = img
    image[:, :, 2] = img
    overlay = overlay.astype(np.uint8)
    image = image.astype(np.uint8)
    print(image.shape, overlay.shape)
    print(type(image), type(overlay))
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    dst = cv2.addWeighted(image, alpha, overlay_bgr, 1 - alpha, 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    return dst


def connected_component(img, connectivity=8):
    binary_map = (img > 127).astype(np.uint8)
    output = cv2.connectedComponentsWithStats(binary_map, connectivity,
                                              cv2.CV_32S)
    stats = output[2]
    df = pd.DataFrame(stats[1:])
    df.columns = ['Left', 'Top', 'Spatial Spread', 'Duration', 'Area']
    df.insert(0, 'Frequency', df.index + 1)
    return df


def load_local_model(weight_name, opt):
    img_shape = (64, 64, 1)
    label_shape = (64, 64, 1)
    x_global = (32, 32, 64)

    weight_files_dir = os.path.join(dirname, 'weight_file/')
    local_weight_filename = weight_files_dir + 'local_model_' + weight_name + '.h5';

    g_local_model = fine_generator(x_global, img_shape)
    g_local_model.load_weights(local_weight_filename)
    g_local_model.compile(loss='mse', optimizer=opt)

    return g_local_model


def load_global_model(weight_name, opt):
    img_shape_g = (32, 32, 1)
    weight_files_dir = os.path.join(dirname, 'weight_file/')
    global_weight_filename = weight_files_dir + 'global_model_' + weight_name + '.h5';
    g_global_model = coarse_generator(img_shape_g, n_downsampling=2, n_blocks=9,
                                      n_channels=1)
    g_global_model.load_weights(global_weight_filename)
    g_global_model.compile(loss='mse', optimizer=opt)

    return g_global_model


def process(input_images, run_dir, run_id, weight_name, stride,
    crop_size, thresh, connectivity, alpha, height_calibration,
    width_calibration):
    # await asyncio.sleep(5)
    K.clear_session()
    gc.collect()

    crop_size_h = crop_size
    crop_size_w = crop_size

    opt = Adam()
    g_local_model = load_local_model(weight_name, opt)
    g_global_model = load_global_model(weight_name, opt)

    global_quant_df = pd.DataFrame(pd.np.empty((0, 8)))
    global_quant_df.columns = ['Image', 'Frequency', 'Left', 'Top', 'Spatial Spread', 'Duration', 'Area', 'Interval']

    global_cal_quant_df = pd.DataFrame(pd.np.empty((0, 8)))
    global_cal_quant_df.columns = ['Image','Frequency', 'Left', 'Top', 'Spatial Spread', 'Duration', 'Area', 'Interval']

    for image_path in input_images:

        img = Image.open(image_path)
        img_arr = np.asarray(img, dtype=np.float32)
        img_arr = img_arr[:, :, 0]
        out_img = strided_crop(img_arr, img_arr.shape[0], img_arr.shape[1],
                               crop_size_h, crop_size_w, g_global_model,
                               g_local_model, stride)
        out_img_sv = out_img.copy()
        out_img_sv = ((out_img_sv) * 255.0).astype('uint8')

        out_img_sv = out_img_sv.astype(np.uint8)
        out_im = Image.fromarray(out_img_sv)
        predicted_image_name = image_path.replace('_original_',
                                                   '_prediction_')

        out_im.save(predicted_image_name)

        out_img_thresh = out_img_sv.copy()
        thresh_img = threshold(out_img_thresh, thresh)
        thresh_im = Image.fromarray(thresh_img)
        threshold_image_name = image_path.replace('_original_',
                                                '_threshold_')
        thresh_im.save(threshold_image_name)

        cc_img = thresh_img.copy()
        df = connected_component(cc_img, connectivity)
        df = interval(df)
        df['Image']=os.path.basename(image_path)
        #df['Image']=' '
        # df.at[0,'Image']=os.path.basename(image_path)
        global_quant_df = global_quant_df.append(df, sort = False)

        df["Duration"] = height_calibration * df["Duration"]
        df["Spatial Spread"] = width_calibration * df["Spatial Spread"]
        df["Area"] = height_calibration * width_calibration * df["Area"]
        df = interval(df)
        # df['Image'] = ' '
        # df[0,'Image']=os.path.basename(image_path)
        global_cal_quant_df = global_cal_quant_df.append(df, sort = False)

        ovleray_img = overlay(img_arr.copy(), thresh_img.copy(), alpha)
        ovleray_im = Image.fromarray(ovleray_img)
        overlay_image_name = image_path.replace('_original_','_overlay_')
        ovleray_im.save(overlay_image_name)

    # stats file
    stats_df = stats.stats(global_cal_quant_df)
    stats_df.to_csv(f'{run_dir}/{run_id}/calibrated_quant_stats.csv', index=False)

    # quant file
    global_quant_df = remove_image_duplicate_name(global_quant_df)
    global_quant_df.to_csv(f'{run_dir}/{run_id}/quant.csv', index=False)

    # calibrated quant file
    global_cal_quant_df = remove_image_duplicate_name(global_cal_quant_df)
    global_cal_quant_df.to_csv(f'{run_dir}/{run_id}/calibrated_quant.csv', index=False)
    #
    def generate_all_groups_plots(run_dir):
        df = None
        for subdir, dirs, files in os.walk(run_dir):
            for dir in dirs:
                stat_file = f'{run_dir}/{dir}/calibrated_quant_stats.csv'
                df1 = pd.read_csv(stat_file)
                df1['category'] = dir
                if df is None:
                    df = df1

                else:
                    df = df.append(df1, sort = False)

        stats.generate_plot_cat(df, y='Spatial Spread_mean', title='Spatial spread',ylabel=r'Distance ($\mu$m)', file_name=f'{run_dir}/spatial_spread.jpg')
        stats.generate_plot_cat(df, y='Area_mean', title='Area', ylabel=r'$\mu$m*s', file_name=f'{run_dir}/area.jpg')
        stats.generate_plot_cat(df, y='Duration_mean', title='Duration', ylabel=r'Time (ms)', file_name=f'{run_dir}/duration.jpg')
        stats.generate_plot_cat(df, y='Interval_mean', title='Interval', ylabel=r'Time (ms)', file_name=f'{run_dir}/interval.jpg')
        stats.generate_plot_cat(df, y='Frequency_count', title='Frequency', ylabel=r'No. of Ca$^{2+}$ Events' +'\n (per STMap)', file_name=f'{run_dir}/frequency.jpg')



    # plots from stats file
    stats_df['category'] = run_id
    stats.generate_plot_cat(stats_df, y='Spatial Spread_mean', title='Spatial spread', ylabel=r'Distance ($\mu$m)', file_name=f'{run_dir}/{run_id}/spatial_spread.jpg')
    stats.generate_plot_cat(stats_df, y='Area_mean', title='Area', ylabel=r'$\mu$m*s', file_name=f'{run_dir}/{run_id}/area.jpg')
    stats.generate_plot_cat(stats_df, y='Duration_mean', title='Duration', ylabel=r'Time (ms)', file_name=f'{run_dir}/{run_id}/duration.jpg')
    stats.generate_plot_cat(stats_df, y='Interval_mean', title='Interval', ylabel=r'Time (ms)', file_name=f'{run_dir}/{run_id}/interval.jpg')
    stats.generate_plot_cat(stats_df, y='Frequency_count', title='Frequency', ylabel=r'No. of Ca$^{2+}$ Events' +'\n (per STMap)', file_name=f'{run_dir}/{run_id}/frequency.jpg')

    generate_all_groups_plots(run_dir)


if __name__ == "__main__":
    process(None, '2021-09-06 05:09:40.722')
