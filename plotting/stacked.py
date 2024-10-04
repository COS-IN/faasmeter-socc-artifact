import os, sys, path
import matplotlib
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import pandas as pd

from plotting.base import PlotBase
from faasmeter.faasmeter.helper_funcs import *
import matplotlib.pyplot as plt

def get_func_name( f ):
    f = f.split('-')[0]
    f = f.split('_')[-2]
    return f

def get_input_size( f ):
    f = f.split('-')[0]
    f = f.split('_')[-1]
    return f

class StackPlot(PlotBase):

    def plot_draw( self, agg, subs, start_zero ):
        agg = agg.copy()
        subs = subs.copy()
        ax = self.axs
        if start_zero:
            start = agg.index[0]
            agg.index -= start
            subs.index -= start
        if 'cp' in subs.columns:
            cols = [ c for c in subs.columns if c != 'cp' ]
            subs = subs[cols]
        labels = [ function_name_to_paper(c) for c in subs.columns ]
        ys = []
        for c in subs.columns:
            ys.append( subs[c] )
        ax.stackplot(
                    subs.index,
                    *ys,
                    labels=labels,
                    baseline='zero',
                    colors=self.get_color_scheme('pulse')
        )
        ax.plot( agg.index, agg, label='System', linestyle="--", color='red' )
        ax.set_xlabel('Time (mins)')
        ax.set_ylabel('Power (Watts)')
        items = (len(subs.columns)+1)
        if items > 5:
            ncols = items//2
        else:
            ncols = 4 
        self.fig.legend(ncol=ncols, bbox_to_anchor=(1.0, 1.0))

    def init_genaxis(self, ):
        plt.close()
        fig, axs = plt.subplots( 
                                self.rows, 
                                self.cols, 
                                figsize=(self.width,self.height), 
                                sharex=False, 
                                sharey=False 
        )
        
        self.fig = fig
        self.axs = axs

        w = 0.2
        self.bar_w = w
        self.bar_offset = w + 0.03
        self.count = 0

    def init_setrcParams(self):
        matplotlib.rcParams.update({'font.size': 12})
        matplotlib.rcParams['pdf.fonttype'] = 42
        # matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
        matplotlib.rcParams['lines.linewidth'] = 1.5
        matplotlib.rcParams["pdf.use14corefonts"] = True
        matplotlib.rcParams['axes.linewidth'] = 0.5
        matplotlib.rcParams['axes.labelpad'] = 2.0
        matplotlib.rcParams['axes.titlepad'] = 2.0
        matplotlib.rcParams['figure.dpi'] = 200.0

        matplotlib.rcParams['figure.subplot.left'] = 0.10  # the left side of the subplots of the figure
        matplotlib.rcParams['figure.subplot.right'] = 0.98    # the right side of the subplots of the figure
        matplotlib.rcParams['figure.subplot.bottom'] = 0.2   # the bottom of the subplots of the figure
        matplotlib.rcParams['figure.subplot.top'] = 0.78      # the top of the subplots of the figure
        matplotlib.rcParams['figure.subplot.wspace'] = 0.25   # the amount of width reserved for blank space between subplots
        matplotlib.rcParams['figure.subplot.hspace'] = 0.2   # the amount of height reserved for white space between subplots
        # matplotlib.rcParams['text.usetex'] = True


if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser( description="Stack plotting for given dir" )
    argparser.add_argument( "--dirs", help="Log Directories", required=False, type=str, nargs='+')                
    args = argparser.parse_args()

    p_ext = '.pickle'

    def load_df( d, name ): 
        print(d+name)
        mc_sys = d+name+p_ext
        df_mc_sys = load_pickle( mc_sys )
        return df_mc_sys
    
    def get_f23( d ):
        df_linear_cpu = load_df(d, '/dfs/cpu/'+tags_pmins[tag_pcpu])
        tag = 'stacked_' + tags_pmins[tag_pshared]
        df_kf_shared = load_df(d, '/dfs/combined/'+tag)
        df_kf_f23 = df_linear_cpu + df_kf_shared
        df_kf_f23 = df_kf_f23.fillna(0)
        return df_kf_f23
    def get_sys( d ):
        df_sys = load_df(d, '/dfs/analysis/all_power')
        return df_sys['igpm']

    dfs_f23 = {}
    dfs_sys = {}
    for d in args.dirs:
        dfs_f23[d] = get_f23( d )
        dfs_sys[d] = get_sys( d )
    
        plot = StackPlot( 1, 1, 5, 3, d + '/dfs/plots/standalone/stacked_f23' )
        plot.plot_init()
        plot.plot_draw( 
                        dfs_sys[d], 
                        dfs_f23[d], 
                        True 
                    )
        plot.plot_close()

