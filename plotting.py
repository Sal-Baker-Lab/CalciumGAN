import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from multiapp import MultiApp

st.title("Hello Bar")
titanic = sns.load_dataset('titanic')
titanic.head()
sns.set_theme()

import pandas as pd
base_dir = '~/research/CalciumGAN/runs/'
input_file_name = '288418_calibrated_quant_S_17_T_6_C_41-13-21_ANO1KD_cntrl_area2_STMapcell1left copy.csv'
df = pd.read_csv(base_dir+input_file_name)


df = df.assign(category='')
fig, ax = plt.subplots() #solved by add this line
ax = sns.swarmplot(x='category', y='Height', data=df, dodge=True, palette='viridis')

# plt.title("Area")
# plt.show()

st.title("Eco")
st.pyplot(fig)




# pd.options.plotting.backend = "plotly"

# import pandas as pd
# base_dir = '~/research/CalciumGAN/runs/'
# input_file_name = '288418_calibrated_quant_S_17_T_6_C_41-13-21_ANO1KD_cntrl_area2_STMapcell1left copy.csv'
# df = pd.read_csv(base_dir+input_file_name)


import plotly.express as px
# # df = px.data.tips()
# fig = px.scatter(x=[0, 0, 0, 0, 0, 0, 0, 0], y=[0, 1, 4, 9, 16, 16, 16, 16])
# fig.show()

