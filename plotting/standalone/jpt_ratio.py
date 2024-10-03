import os, sys, path
import matplotlib

paths = [
    # "/data2/ar/iluvatar-energy-experiments/scripts/experiments/victor/plotting",
    # "/data2/ar/iluvatar-energy-experiments/faasmeter",
    "/data2/ar/faasmeter/faasmeter",
    "/data2/ar/iluvatar-energy-experiments/scripts/experiments/desktop/native_func_stuff/native_func",
]

for p in paths:
    current = path.Path( os.path.realpath( p ) ).abspath()
    sys.path.append( current )
    print( "added {} to PATH".format( current ) )

import numpy as np
import pandas as pd

from plotting.base import PlotBase
from helper_funcs import *
import matplotlib.pyplot as plt

def get_func_name( f ):
    f = f.split('-')[0]
    f = f.split('_')[-2]
    return f

def get_input_size( f ):
    f = f.split('-')[0]
    f = f.split('_')[-1]
    return f

class MCBarPlot(PlotBase):

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
        matplotlib.rcParams.update({'font.size': 10})
        matplotlib.rcParams['pdf.fonttype'] = 42
        # matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
        matplotlib.rcParams['lines.linewidth'] = 1.5
        matplotlib.rcParams["pdf.use14corefonts"] = True
        matplotlib.rcParams['axes.linewidth'] = 0.5
        matplotlib.rcParams['axes.labelpad'] = 2.0
        matplotlib.rcParams['axes.titlepad'] = 2.0
        matplotlib.rcParams['figure.dpi'] = 200.0

        matplotlib.rcParams['figure.subplot.left'] = 0.20  # the left side of the subplots of the figure
        matplotlib.rcParams['figure.subplot.right'] = 0.95    # the right side of the subplots of the figure
        matplotlib.rcParams['figure.subplot.bottom'] = 0.25   # the bottom of the subplots of the figure
        matplotlib.rcParams['figure.subplot.top'] = 0.9      # the top of the subplots of the figure
        matplotlib.rcParams['figure.subplot.wspace'] = 0.25   # the amount of width reserved for blank space between subplots
        matplotlib.rcParams['figure.subplot.hspace'] = 0.2   # the amount of height reserved for white space between subplots
        # matplotlib.rcParams['text.usetex'] = True

    def plot_draw( self, ratio_mean, ratio_std, ylabel ):
        
        ax = self.axs

        x = [i for i in range(0,len(ratio_mean))]
       
        ax.errorbar( x, ratio_mean, yerr=ratio_std )

        ax.set_ylim(0)
        
        # rotation=0,
        ax.set_ylabel( ylabel,  labelpad=5.0, loc='center')
        ax.set_xlabel( r'$Experiment\ Number$' )

if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser( description="Ratio plotting across exps " )
    argparser.add_argument( "--mc_a_dirs", '-s', help="Log Directories for mc_a exps of the three platforms", required=True, type=str, nargs='+')                
    argparser.add_argument( "--platform", help="Name of the platform", required=True, type=str)                
    args = argparser.parse_args()

    p_ext = '.pickle'
    
    mc_a_dirs = args.mc_a_dirs

    def load_df( d, name ): 
        print(d+name)
        mc_sys = d+name+p_ext
        df_mc_sys = load_pickle( mc_sys )
        return df_mc_sys

    ratio_mean = []
    ratio_std = []

    for i, d in enumerate(mc_a_dirs):
        def try_fill( ls, name, transformer ):
            try: 
                df = load_df(d, '/dfs/analysis/'+name)
                ls.append( transformer(df) )
            except FileNotFoundError:
                ls.append( 0.0 )
        def get_mean( df ):
            return df.mean()
        def get_std( df ):
            return df.std()
        try_fill( ratio_mean, 'ratio_jpt_full', get_mean )
        try_fill( ratio_std, 'ratio_jpt_full', get_std )
        
    def sort_by_mean_std(element):
        idx = ratio_std.index(element)
        return ratio_mean[idx]
    def sort_by_mean_idx(element):
        idx = mc_a_dirs.index(element)
        return ratio_mean[idx]
    ratio_std = sorted(ratio_std, key=sort_by_mean_std)
    mc_a_dirs = sorted(mc_a_dirs, key=sort_by_mean_idx)
    ratio_mean = sorted(ratio_mean)

    for i, d in enumerate(mc_a_dirs):
        print("{},{},{} - {}".format(i, ratio_mean[i], ratio_std[i], d))

    base_dir = mc_a_dirs[0]

    plot = MCBarPlot( 1, 1, 5, 2.0, base_dir + '/dfs/plots/standalone/jpt_ratio_'+args.platform )
    plot.plot_init()
    plot.plot_draw( 
                    ratio_mean, 
                    ratio_std,
                    r'$\frac{\sigma(\ J\ per\ Invoke\ )}{\sigma(\ latency\ )}$' 
                  )
    plot.plot_close()

