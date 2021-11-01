# CalciumGAN

This code is part of our paper "CalciumGAN: Segmenting and quantifying calcium signals using multi-scale generative adversarial networks". 

The code is authored and maintained by Sharif Amit Kamran [[Webpage]](https://www.sharifamit.com/) and Hussein Moghnieh [[Webpage]](https://medium.com/@husseinmoghnie).

## Pre-requisite
- Ubuntu 18.04 / Windows 7 or later
- NVIDIA Graphics card


## Installation Instruction for Ubuntu
- Download and Install [Nvidia Drivers](https://www.nvidia.com/Download/driverResults.aspx/142567/en-us)
- Download and Install via Runfile [Nvidia Cuda Toolkit 10.0](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)
- Download and Install [Nvidia CuDNN 7.6.5 or later](https://developer.nvidia.com/rdp/cudnn-archive)
- Install Pip3 and Python3 enviornment

```
sudo apt-get install pip3 python3-dev
```
- Install Tensorflow-Gpu version-2.0.0 and Keras version-2.3.1
```
sudo pip3 install tensorflow-gpu==2.0.3
sudo pip3 install keras==2.3.1
```
- Install packages from requirements.txt
```
sudo pip3 install -r requirements.txt
```

## Instructions for running the code

- Put the original images in 'test/JPEGImages' folder.
- Put the weight files in 'weight_file' folder.
- Run the code in command line given below.
- [Optional] For best result use stride=3.

## Testing on Calcium Images

- Type this in terminal to run the infer.py file
```
python3 infer.py 
```

- There are different flags to choose from. Not all of them are mandatory

```
    '--in_dir', type=str, default='test', help='path/to/save/dir'
    '--weight_name', type=str, default='test', help='.h5 file name'   
    '--stride', type=int, default=3
    '--crop_size', type=int, default=64
```


# License

The code is released under the GPL-2 License, you can read the license file included in the repository for details.
