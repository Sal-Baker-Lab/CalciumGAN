import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import random
import numpy

def stats(global_quant_df):
    stats_df = global_quant_df.groupby(by=["Image"]).agg(['mean', 'count'])
    stats_df = stats_df.reset_index()
    stats_df.columns = [c[0] + "_" + c[1] for c in stats_df.columns]
    #stats_df.drop('Frequency_mean', axis=1, inplace=True)
   # stats_df.drop('Height_count', axis=1, inplace=True)
   # stats_df.drop('Width_count', axis=1, inplace=True)
   # stats_df.drop('Area_count', axis=1, inplace=True)
    return stats_df

def generate_frequency_plot(df, file_name=None):
    df['category'] = numpy.random.uniform(0, 1, len(df))
    fig, ax = plt.subplots()
    ax.figure.set_size_inches(1, 3)

    g = sns.swarmplot(x='category', y='Frequency_count', data=df, dodge=True, palette='viridis', ax=ax)
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)

    ax.set_xlabel('')
    g.set(xticks=[])
    ax.set_ylabel(r'Frequency No. of ' + r'$Ca^2+ Events$' +'\n (per STMap)', fontsize = 18)
    plt.title("Events", fontsize=18)
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

def generate_duration_plot(df, file_name=None):
    df['category'] = numpy.random.uniform(0, 1, len(df))
    fig, ax = plt.subplots()
    ax.figure.set_size_inches(1, 3)

    g = sns.swarmplot(x='category', y='Width_mean', data=df, dodge=True, palette='viridis', ax=ax)
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)

    ax.set_xlabel('')
    g.set(xticks=[])
    ax.set_ylabel(r'Time ($\mu$s)', fontsize = 18)
    plt.title("Duration", fontsize=18)
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()



def generate_area_plot(df, file_name = None):

    df['category'] = numpy.random.uniform(0, 1, len(df))
    fig, ax = plt.subplots()
    ax.figure.set_size_inches(1, 3)

    g = sns.swarmplot(x='category', y='Area_mean', data=df, dodge=True, palette='viridis', ax=ax)
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)

    ax.set_xlabel('')
    g.set(xticks=[])
    ax.set_ylabel(r'Area ($\mu$m*s)', fontsize = 18)
    plt.title("Area", fontsize=18)
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

def generate_interval_plot(df, file_name):
    df['category'] = numpy.random.uniform(0, 1, len(df))
    fig, ax = plt.subplots()
    ax.figure.set_size_inches(1, 3)

    g = sns.swarmplot(x='category', y='Interval_mean', data=df, dodge=True, palette='viridis', ax=ax)
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)

    ax.set_xlabel('')
    g.set(xticks=[])
    ax.set_ylabel(r'Distance \n' + r'$(mu*s)$', fontsize = 18)
    plt.title("Spatial spread", fontsize=18)
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

# df = pd.read_csv('/Users/hussein/research/CalciumGAN/runs/723780/quant.csv')

#print(type(df))
#stats_df = stats(df)

#print(stats_df)
# generate_height_plot(stats_df)
# generate_frequency_plot(stats_df)
# alt.renderers.enable('altair_viewer')

# generate_area_plot(df)
# print('test')
# generate_interval_plot(stats_df)



