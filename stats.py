import matplotlib.pyplot as plt
import numpy
import seaborn as sns
import pandas as pd
import os
PLOT_TITLE_FONT_S=12
PLOT_LABEL_FONT_S=10


def size_plot(plt, scale = 1):
    N = scale
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

def generate_plot_cat(df, y, title, ylabel, file_name):
    scale = 1 if len(df['category'].unique()) == 1 else len(df['category'].unique()) + 1
    ax = sns.catplot(x="category", y=y, kind="swarm", data=df, s=13, marker="$\circ$")
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    size_plot(plt, scale)
    if len(df) == 1:
        ax.set(xticks=[])
    ax.set_xticklabels(rotation = 30)
    ax.set( xlabel = "", ylabel = ylabel)
    plt.title(title, fontsize=PLOT_TITLE_FONT_S)

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')
    else:
        plt.show(bbox_inches='tight')





# generate_all_groups_plots('/Users/hussein/research/CalciumGAN/runs')
#df = pd.read_csv('/Users/hussein/research/CalciumGAN/runs/GanCalcium_run_753652/calibrated_quant_stats.csv')
#df['category']='111'
#generate_plot_cat(df, y='Interval_mean', title='Spatial Spread', ylabel=r'$(mu*s)$', file_name='spatial_spread.jpg')




