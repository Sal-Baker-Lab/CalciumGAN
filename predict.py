import asyncio
import numpy as np
from model import fine_generator, coarse_generator
#from libtiff import TIFF
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter
import random
import cv2
from functools import partial
import numpy as np
import tensorflow as tf
import keras
import argparse
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Model,load_model
import keras.backend as K
from keras.initializers import RandomNormal
from numpy import load
from sklearn.metrics import confusion_matrix,jaccard_similarity_score,f1_score,roc_auc_score,auc,recall_score, auc,roc_curve
import gc
import glob
import pycm

import warnings
warnings.filterwarnings('ignore')

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True


dirname = os.path.dirname(__file__)


def load_local_model(weight_name, opt):
    img_shape = (64,64,1)
    label_shape = (64,64,1)
    x_global = (32,32,64)

    weight_files_dir = os.path.join(dirname, 'weight_file/')
    local_weight_filename = weight_files_dir + 'local_model_'+weight_name+'.h5';

    g_local_model = fine_generator(x_global,img_shape)
    g_local_model.load_weights(local_weight_filename)
    g_local_model.compile(loss='mse', optimizer=opt)

    return g_local_model

def load_global_model(weight_name, opt):
    img_shape_g = (32,32,1)
    weight_files_dir = os.path.join(dirname, 'weight_file/')
    global_weight_filename = weight_files_dir + 'global_model_'+weight_name+'.h5';
    g_global_model = coarse_generator(img_shape_g,n_downsampling=2, n_blocks=9, n_channels=1)
    g_global_model.load_weights(global_weight_filename)
    g_global_model.compile(loss='mse',optimizer=opt)

    return g_global_model


def process(input_image, run_directory, weight_name='000090', stride=3, crop_size=64, threshold=50, connectivity=8):

    # await asyncio.sleep(5)
    K.clear_session()
    gc.collect()

    opt = Adam()
    local_model = load_local_model(weight_name, opt)
    global_model = load_global_model(weight_name, opt)

    # pred_image_path = os.path.join(dirname, 'runs/' + run_directory +'/pred_image.jpg')
    # img.save(pred_image_path)
    # threshold_image_path = os.path.join(dirname, 'runs/' + run_directory +'/thresh_image.jpg')
    # img.save(threshold_image_path)
    # quant_csv_path = os.path.join(dirname, 'runs/' + run_directory +'/quant_csv.csv')

    img_name_path = os.path.join(dirname, 'runs/' + run_directory +'/input_image.jpg')
    img = Image.open(img_name_path)
    img_arr = np.asarray(img,dtype=np.float32)
    img_arr = img_arr[:,:,0]
    out_img = strided_crop(img_arr, img_arr.shape[0], img_arr.shape[1], crop_size_h, crop_size_w,g_global_model,g_local_model,stride)
    out_img_sv = out_img.copy()
    out_img_sv = ((out_img_sv) * 255.0).astype('uint8')

    out_img_sv = out_img_sv.astype(np.uint8)
    out_im = Image.fromarray(out_img_sv)
    pred_image_path = os.path.join(dirname, 'runs/' + run_directory +'/pred_image.jpg')
    out_im.save(pred_image_path)


    out_img_thresh = out_img_sv.copy()
    thresh_img = threshold(out_img_thresh,thresh)
    thresh_im = Image.fromarray(thresh_img)
    threshold_image_path = os.path.join(dirname, 'runs/' + run_directory +'/thresh_image.jpg')
    thresh_im.save(threshold_image_path)

    cc_img = thresh_img.copy()
    df = connected_component(cc_img,connectivity)
    quant_csv_path = os.path.join(dirname, 'runs/' + run_directory +'/quant_csv.csv')
    df.to_csv(quant_csv_path)




