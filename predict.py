import os
import gc
import warnings
warnings.filterwarnings('ignore')

dirname = os.path.dirname(__file__)


def process(input_image, weight_name='000090', stride=3, crop_size=64, threshold=50, connectivity=8):
    K.clear_session()
    gc.collect()

    img_shape = (64,64,1)
    label_shape = (64,64,1)
    x_global = (32,32,64)
    opt = Adam()

    # loading local model
    g_local_model = fine_generator(x_global,img_shape)
    g_local_model.load_weights('weight_file/local_model_'+weight_name+'.h5')
    g_local_model.compile(loss='mse', optimizer=opt)

    # loading global model
    g_global_model = coarse_generator(img_shape_g,n_downsampling=2, n_blocks=9, n_channels=1)
    g_global_model.load_weights('weight_file/global_model_'+weight_name+'.h5')
    g_global_model.compile(loss='mse',optimizer=opt)



