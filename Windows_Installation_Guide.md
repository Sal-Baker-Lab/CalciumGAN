# Installation 

## Pre-requisite
- Windows 7 or later
- **Supports** : NVIDIA Pascal (P100, GTX10**), Volta (V100), Turing (GTX 16**, RTX 20**, Quadro)
- **Does not support** : NVIDIA Amphere (RTX 30**, A100) [In Development]

## 1. Download and Install Anaconda from the following link

```
https://www.anaconda.com/products/individual
```

## 2. Open Anaconda Prompt from Start menu. 

## 3. Type following in terminal to create a virtual environment with anaconda packages
```
create -n streamlit-calciumgan anaconda python=3.7
```
## 3. Activate the virual environment
```
conda activate streamlit-calciumgan
```
## 4. Install tensorflow-gpu and downgrade tensorflow-estimator to 2.0.0
```
conda install tensorflow-gpu=2.0.0
pip install tensorflow-estimator==2.0.0
```

## 5. Clone the repository from github
```
git clone https://github.com/SharifAmit/CalciumGAN.git
```

## 6. Install packages
```
pip install -r requirements
```

## 8. Replace scikit-learn with a downgraded version
```
pip uninstall scikit-learn
pip install scikit-learn==0.20.4
```
## 9. install pycm
```
pip install pycm
```

## 10. Deactivate the virual environment
```
conda deactivate streamlit-calciumgan
```

# Running the app

## 1. Open Anaconda Prompt from Start menu 

## 2. Activate the virual environment
```
source activate streamlit-calciumgan
```
## 4. Go inside the directory and type 
```
streamlit run web_streamlit.py
```
## 5. A new browser will open with CalciumGAN app running on it. 

