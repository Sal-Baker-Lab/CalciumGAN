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

# import keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True


dirname = os.path.dirname(__file__)

def normalize_pred(img,g_global_model,g_local_model):
    img = np.reshape(img,[1,64,64,1])
    img_coarse = tf.image.resize(img, (32,32), method=tf.image.ResizeMethod.LANCZOS3)
    img_coarse = (img_coarse - 127.5) / 127.5
    img_coarse = np.array(img_coarse)

    X_fakeB_coarse,x_global = g_global_model.predict(img_coarse)
    X_fakeB_coarse = (X_fakeB_coarse+1)/2.0
    pred_img_coarse = X_fakeB_coarse[:,:,:,0]


    img = (img - 127.5) / 127.5
    X_fakeB = g_local_model.predict([img,x_global])
    X_fakeB = (X_fakeB+1)/2.0
    pred_img = X_fakeB[:,:,:,0]
    return [np.asarray(pred_img,dtype=np.float32),np.asarray(pred_img_coarse,dtype=np.float32)]

def strided_crop(img, img_h,img_w,height, width,g_global_model,g_local_model,stride=1):

    full_prob = np.zeros((img_h, img_w),dtype=np.float32)
    full_sum = np.ones((img_h, img_w),dtype=np.float32)

    max_x = int(((img.shape[0]-height)/stride)+1)
    max_y = int(((img.shape[1]-width)/stride)+1)
    max_crops = (max_x)*(max_y)
    i = 0
    for h in range(max_x):
        for w in range(max_y):
            crop_img_arr = img[h * stride:(h * stride) + height,w * stride:(w * stride) + width]
            [pred,pred_256] = normalize_pred(crop_img_arr,g_global_model,g_local_model)
            full_prob[h * stride:(h * stride) + height,w * stride:(w * stride) + width] += pred[0]
            full_sum[h * stride:(h * stride) + height,w * stride:(w * stride) + width] += 1
            i = i + 1
            #print(i)
    out_img = full_prob / full_sum
    return out_img

def threshold(img,thresh):

    binary_map = (img > thresh).astype(np.uint8)
    binary_map[binary_map==1] = 255
    return binary_map

def overlay(img,mask,alpha=0.7):

    overlay = np.zeros((mask.shape[0],mask.shape[1],3))
    overlay[mask==255] = 255
    overlay[:,:,1] = 0
    overlay[:,:,2] = 0
    overlay = overlay.astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    dst = cv2.addWeighted(img, alpha, overlay_bgr, 1-alpha, 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    return dst

def connected_component(img,connectivity=8):

    binary_map = (img > 127).astype(np.uint8)
    output = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)
    stats = output[2]
    df = pd.DataFrame(stats[1:])
    df.columns = ['Left','Top','Width','Height','Area']
    return df



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

# max (stride)  = width - crop size

# crop_size = 64  (trained on)
# threshold 1 and 254 (default 32)
#stride start 3, max = width - crop_szie
# connectivity 4, and 8

# view xls v download

# progress bar


def process(input_image, run_dir, original_image_name, weight_name='000090', stride=16, crop_size=64, thresh=50, connectivity=8, alpha=0.7):

    # await asyncio.sleep(5)
    K.clear_session()
    gc.collect()

    crop_size_h = crop_size
    crop_size_w = crop_size

    opt = Adam()
    g_local_model = load_local_model(weight_name, opt)
    g_global_model = load_global_model(weight_name, opt)


    img_name_path = os.path.join(dirname, 'runs/' + original_image_name)
    img = Image.open(img_name_path)
    img_arr = np.asarray(img,dtype=np.float32)
    img_arr = img_arr[:,:,0]
    out_img = strided_crop(img_arr, img_arr.shape[0], img_arr.shape[1], crop_size_h, crop_size_w, g_global_model, g_local_model, stride)
    out_img_sv = out_img.copy()
    out_img_sv = ((out_img_sv) * 255.0).astype('uint8')

    out_img_sv = out_img_sv.astype(np.uint8)
    out_im = Image.fromarray(out_img_sv)
    pred_image_path = os.path.join(dirname, 'runs/' + original_image_name.replace('_original_', '_prediction_'))
    out_im.save(pred_image_path)


    out_img_thresh = out_img_sv.copy()
    thresh_img = threshold(out_img_thresh, thresh)
    thresh_im = Image.fromarray(thresh_img)
    threshold_image_path =  os.path.join(dirname, 'runs/' + original_image_name.replace('_original_', '_threshold_'))
    thresh_im.save(threshold_image_path)

    cc_img = thresh_img.copy()
    df = connected_component(cc_img,connectivity)
    quant_csv_path =  os.path.join(dirname, 'runs/' + original_image_name.replace('_original_', '_quant_'))
    quant_csv_path = quant_csv_path.replace('jpg', 'csv')
    df.to_csv(quant_csv_path)

    ovleray_img = overlay(out_img_sv.copy(), thresh_img.copy(), alpha)
    ovleray_im = Image.fromarray(ovleray_img)
    overlay_image_path =  os.path.join(dirname, 'runs/' + original_image_name.replace('_original_', '_overlay_'))
    ovleray_im.save(overlay_image_path)

if __name__ == "__main__":
    process(None, '2021-09-06 05:09:40.722')



