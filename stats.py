import matplotlib.pyplot as plt
import numpy
import seaborn as sns
import pandas as pd

PLOT_TITLE_FONT_S=12
PLOT_LABEL_FONT_S=10


def size_plot(plt):
    N = 1
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2 # inch margin
    s = maxsize/plt.gcf().dpi*N+2*m
    margin = m/plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1.-margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])


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

    g = sns.swarmplot(x='category', y='Frequency_count', data=df, dodge=True, palette='viridis', ax=ax)
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    size_plot(plt)

    ax.set_xlabel('')
    g.set(xticks=[])
    ax.set_ylabel(r'Frequency No. of ' + r'$Ca^2+ Events$' +'\n (per STMap)', fontsize = PLOT_LABEL_FONT_S)
    plt.title("Events", fontsize=PLOT_TITLE_FONT_S)
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

def generate_duration_plot(df, file_name=None):
    df['category'] = numpy.random.uniform(0, 1, len(df))
    fig, ax = plt.subplots()

    g = sns.swarmplot(x='category', y='Width_mean', data=df, dodge=True, palette='viridis', ax=ax)
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    size_plot(plt)

    ax.set_xlabel('')
    g.set(xticks=[])
    ax.set_ylabel(r'Time ($\mu$s)', fontsize = PLOT_LABEL_FONT_S)
    plt.title("Duration", fontsize=PLOT_TITLE_FONT_S)
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

def generate_area_plot(df, file_name = None):

    df['category'] = numpy.random.uniform(0, 1, len(df))
    fig, ax = plt.subplots()

    g = sns.swarmplot(x='category', y='Area_mean', data=df, dodge=True, palette='viridis', ax=ax)
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    size_plot(plt)

    ax.set_xlabel('')
    g.set(xticks=[])
    ax.set_ylabel(r'Area ($\mu$m*s)', fontsize = PLOT_LABEL_FONT_S)
    plt.title("Area", fontsize=PLOT_TITLE_FONT_S)
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

def generate_interval_plot(df, file_name):
    df['category'] = numpy.random.uniform(0, 1, len(df))
    fig, ax = plt.subplots()

    g = sns.swarmplot(x='category', y='Interval_mean', data=df, dodge=True, palette='viridis', ax=ax)
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    size_plot(plt)

    ax.set_xlabel('')
    g.set(xticks=[])
    ax.set_ylabel(r'Distance \n' + r'$(mu*s)$', fontsize = PLOT_LABEL_FONT_S)
    plt.title("Spatial spread", fontsize=PLOT_TITLE_FONT_S)

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

df = pd.read_csv('/Users/hussein/research/CalciumGAN/runs/723780/quant_stats.csv')
#
generate_interval_plot(df, file_name=None)



