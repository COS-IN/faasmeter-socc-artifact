import os, sys, path
import matplotlib
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
        matplotlib.rcParams.update({'font.size': 14})
        matplotlib.rcParams['pdf.fonttype'] = 42
        # matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
        matplotlib.rcParams['lines.linewidth'] = 1.5
        matplotlib.rcParams["pdf.use14corefonts"] = True
        matplotlib.rcParams['axes.linewidth'] = 0.5
        matplotlib.rcParams['axes.labelpad'] = 2.0
        matplotlib.rcParams['axes.titlepad'] = 2.0
        matplotlib.rcParams['figure.dpi'] = 200.0

        matplotlib.rcParams['figure.subplot.left'] = 0.1  # the left side of the subplots of the figure
        matplotlib.rcParams['figure.subplot.right'] = 0.95    # the right side of the subplots of the figure
        matplotlib.rcParams['figure.subplot.bottom'] = 0.15   # the bottom of the subplots of the figure
        matplotlib.rcParams['figure.subplot.top'] = 0.95      # the top of the subplots of the figure
        matplotlib.rcParams['figure.subplot.wspace'] = 0.25   # the amount of width reserved for blank space between subplots
        matplotlib.rcParams['figure.subplot.hspace'] = 0.2   # the amount of height reserved for white space between subplots
        # matplotlib.rcParams['text.usetex'] = True

    def plot_draw( self, error_mean, error_std, ylabel ):
        from matplotlib.ticker import MaxNLocator
        
        ax = self.axs

        x = [i for i in range(0,len(error_mean))]
        # print(error_mean)
        #ax.errorbar( x, error_mean, yerr=error_std )
        ax.plot( x, error_mean, marker='.' )

        ax.set_ylim(0)
        ax.set_xlim(0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # rotation=0,
        ax.set_ylabel( ylabel,  labelpad=5.0, loc='center')
        ax.set_xlabel( 'Trace ID' )
        ax.grid()

if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser( description="error plotting across exps " )
    argparser.add_argument( "--mc_a_dirs", '-s', help="Log Directories for mc_a exps of the three platforms", required=True, type=str, nargs='+')                
    argparser.add_argument( "--platform", help="Name of the platform", required=True, type=str)                
    args = argparser.parse_args()
    # print(args)
    # exit()
    p_ext = '.pickle'
    
    mc_a_dirs = args.mc_a_dirs

    def load_df( d, name ): 
        print(d+name)
        mc_sys = d+name+p_ext
        df_mc_sys = load_pickle( mc_sys )
        return df_mc_sys

    error_mean = []
    error_std = []

    for i, d in enumerate(mc_a_dirs):
        def try_fill( ls, name, transformer ):
            try: 
                df = load_df(d, '/dfs/analysis/'+name)
                ls.append( transformer(df) )
            except FileNotFoundError:
                ls.append( 0.0 )
        def get_mean( df ):
            return np.abs(df.mean())
        def get_std( df ):
            return df.std()
        try_fill( error_mean, 'sys_full_error_mins', get_mean )
        try_fill( error_std, 'sys_full_error_mins', get_std )
        
    def sort_by_mean_std(element):
        idx = error_std.index(element)
        return error_mean[idx]
    def sort_by_mean_idx(element):
        idx = mc_a_dirs.index(element)
        return error_mean[idx]
    error_std = sorted(error_std, key=sort_by_mean_std)
    mc_a_dirs = sorted(mc_a_dirs, key=sort_by_mean_idx)
    error_mean = sorted(error_mean)

    print("Total Experiments: {}".format(len(error_mean)))

    i = list(reversed(error_mean)).index( 0.0 )
    idx_s = len(error_mean) - i
    idx_e = next(x[0] for x in enumerate(error_mean) if x[1] > 20.0) 

    error_mean = error_mean[idx_s:idx_e]
    error_std = error_std[idx_s:idx_e]
    mc_a_dirs = mc_a_dirs[idx_s:idx_e]

    for i, d in enumerate(mc_a_dirs):
        print("{},{},{} - {}".format(i, error_mean[i], error_std[i], d))

    base_dir = mc_a_dirs[0]

    plot = MCBarPlot( 1, 1, 5, 3.0, './jpt_error_'+args.platform )
    plot.plot_init()
    plot.plot_draw( 
                    error_mean, 
                    error_std,
                    'Total-Error %' 
                  )
    plot.plot_close()

