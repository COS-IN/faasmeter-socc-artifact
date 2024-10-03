import os, sys, path
import matplotlib
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

paths = [
    # "/data2/ar/iluvatar-energy-experiments/scripts/experiments/victor/plotting",
    # "/data2/ar/iluvatar-energy-experiments/faasmeter",
    # "/data2/ar/faasmeter/faasmeter",
    # "/data2/ar/iluvatar-energy-experiments/scripts/experiments/desktop/native_func_stuff/native_func",
    # "/extra/alfuerst/repos/faasmeter-open/faasmeter",
    # "/extra/alfuerst/repos/faasmeter/faasmeter",
    # "/extra/alfuerst/repos/iluvatar-energy-experiments/scripts/experiments/desktop/native_func_stuff/native_func",
]

for p in paths:
    current = path.Path( os.path.realpath( p ) ).abspath()
    sys.path.append( current )
    print( "added {} to PATH".format( current ) )

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
        matplotlib.rcParams['figure.subplot.bottom'] = 0.2   # the bottom of the subplots of the figure
        matplotlib.rcParams['figure.subplot.top'] = 0.8      # the top of the subplots of the figure
        matplotlib.rcParams['figure.subplot.wspace'] = 0.40   # the amount of width reserved for blank space between subplots
        matplotlib.rcParams['figure.subplot.hspace'] = 0.2   # the amount of height reserved for white space between subplots
        # matplotlib.rcParams['text.usetex'] = True

    def plot_draw( self, 
                    desk_mean_full,
                    server_mean_full,
                    jetson_mean_full,
                    desk_mean_norm,
                    server_mean_norm,
                    jetson_mean_norm,
                 ):
        
        fig = self.fig
        
        def plot_on_ax( ax, desk_vec, server_vec, jetson_vec ):
            minx = min( desk_vec + server_vec + jetson_vec )
            maxx = max( desk_vec + server_vec + jetson_vec )

            x = [ x for x in np.arange(0,maxx,maxx/100) ]
            
            def get_ys( vec ):
                ecdf = ECDF(vec)
                return ecdf( x )
            
            y_desk = get_ys( desk_vec )
            y_server = get_ys( server_vec )
            y_jetson = get_ys( jetson_vec )
            
            colors_map = [
                'black',
                'tab:red',
                'tab:blue',
            ]
            ax.plot( x, y_desk, label=gen_label('Desktop'), color=colors_map[0] )
            ax.plot( x, y_server, label=gen_label('Server'), color=colors_map[1] )
            ax.plot( x, y_jetson, label=gen_label('Jetson'), color=colors_map[2] )

            ax.set_ylim(0)
            ax.yaxis.set_minor_locator(MultipleLocator(0.1))
            if maxx > 10:
                ax.xaxis.set_minor_locator(MultipleLocator(20))
                ax.xaxis.set_major_locator(MultipleLocator(40))
            else:
                ax.xaxis.set_minor_locator(MultipleLocator(0.1))
                ax.xaxis.set_major_locator(MultipleLocator(0.4))
            ax.set_ylabel( r'CDF',  labelpad=5.0, loc='center')
            ax.set_xlabel( ylabel )
        
        def gen_label(l):
            if do_gen:
                return l
            return None
        do_gen = True
        ylabel = r'$\sigma(J)\ /\ \sigma(T)$' 
        plot_on_ax( self.axs[0], desk_mean_full, server_mean_full, jetson_mean_full )

        do_gen = False 
        ylabel = r'$\sigma(J)\ /\ E[J]$' 
        plot_on_ax( self.axs[1], desk_mean_norm, server_mean_norm, jetson_mean_norm )

        fig.legend(  bbox_to_anchor=(0.85, 0.95), ncols=3)
        

if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser( description="Ratio plotting across exps " )
    argparser.add_argument( "--dirs_desktop", help="Log Directories", required=True, type=str, nargs='+')                
    argparser.add_argument( "--dirs_server", help="Log Directories", required=True, type=str, nargs='+')                
    argparser.add_argument( "--dirs_jetson", help="Log Directories", required=True, type=str, nargs='+')                
    argparser.add_argument( "--platform", help="Name of the platform", required=True, type=str)                
    args = argparser.parse_args()

    p_ext = '.pickle'

    def load_df( d, name ): 
        print(d+name)
        mc_sys = d+name+p_ext
        df_mc_sys = load_pickle( mc_sys )
        return df_mc_sys
    
    def get_vecs( dirs, name ):
        ratio_mean = []
        ratio_vec = []
        ratio_std = []

        for i, d in enumerate(dirs):
            def try_fill( ls, name, transformer ):
                try: 
                    df = load_df(d, '/dfs/analysis/'+name)
                    df = transformer(df) 
                    if isinstance( df, list ):
                        ls.extend( df )
                    else:
                        ls.append( df )
                except FileNotFoundError:
                    ls.append( 0.0 )
            def get_vec( df ):
                vec = df.to_numpy()
                return list(vec) 
            def get_mean( df ):
                return df.mean()
            def get_std( df ):
                return df.std()
            try_fill( ratio_vec, name, get_vec )
            try_fill( ratio_mean, name, get_mean )
            try_fill( ratio_std, name, get_std )
        return ratio_vec, ratio_mean, ratio_std

    desk_vec_full, desk_mean_full, desk_std_full = get_vecs( args.dirs_desktop, 'ratio_jpt_full' )
    server_vec_full, server_mean_full, server_std_full = get_vecs( args.dirs_server, 'ratio_jpt_full'  )
    jetson_vec_full, jetson_mean_full, jetson_std_full = get_vecs( args.dirs_jetson, 'ratio_jpt_full'  )

    desk_vec_norm, desk_mean_norm, desk_std_norm = get_vecs( args.dirs_desktop, 'jpt_full_std_normalized' )
    server_vec_norm, server_mean_norm, server_std_norm = get_vecs( args.dirs_server, 'jpt_full_std_normalized'  )
    jetson_vec_norm, jetson_mean_norm, jetson_std_norm = get_vecs( args.dirs_jetson, 'jpt_full_std_normalized'  )

    base_dir = args.dirs_desktop[0]

    plot = MCBarPlot( 1, 2, 6.0, 2.5, base_dir + '/dfs/plots/standalone/jpt_ratio_cdf_single'+args.platform )
    plot.plot_init()
    plot.plot_draw( 
                    desk_mean_full,
                    server_mean_full,
                    jetson_mean_full,
                    desk_vec_norm,
                    server_vec_norm,
                    jetson_vec_norm,
                  )
    plot.plot_close()



