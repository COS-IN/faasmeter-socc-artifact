import os, sys, path

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
    return int(f)

class MCBarPlot(PlotBase):

    def plot_draw( self, mc_sys, title ):
        funcs = mc_sys.columns
        funcs_p = [ function_name_to_paper(f) for f in funcs ]
        funcs_size = [ get_input_size(f) for f in funcs ]
        funcs_name = [ get_func_name(f) for f in funcs ]
        funcs_name = set(funcs_name)
        if len(funcs_name) != 1:
            print("Error: various input sizes experiment has more then one type of functions!")

        container = []

        for f in funcs:
            fn = f.split('-')[0]
            container.append( mc_sys[f][0] )

        w = 0.4
        offset = w + 0.05
        count = 1
        ax = self.axs

        # ax.bar( funcs_size, container, width=w, label='Pyaes' )
        ax.plot( funcs_size, container, marker='o' )
        
        self.fig.legend(bbox_to_anchor=(1.01, 1.0), ncol=1)
        ax.set_ylabel( 'Energy per invocation (J)' )
        ax.set_xlabel( 'Input Parameter: Number of Iterations')
        ax.set_xlim(0)
        ax.set_ylim(0)
        # plt.xticks(rotation='vertical')
        self.fig.suptitle( title )

class ExecBarPlot(PlotBase):

    def plot_draw( self, exec_time, title ):
        funcs = exec_time.index
        funcs_p = [ function_name_to_paper(f) for f in funcs ]
        funcs_size = [ get_input_size(f) for f in funcs ]
        funcs_name = [ get_func_name(f) for f in funcs ]
        funcs_name = set(funcs_name)
        if len(funcs_name) != 1:
            print("Error: various input sizes experiment has more then one type of functions!")

        container = []

        for f in funcs:
            fn = f.split('-')[0]
            container.append( exec_time.loc[f]['exec_time']['mean'] )
        
        w = 0.4
        offset = w + 0.05
        count = 1
        ax = self.axs

        ax.plot( funcs_size, container, marker='o' )
        
        # plt.xticks(rotation='vertical')
        self.fig.legend(bbox_to_anchor=(1.01, 1.0), ncol=1)
        ax.set_ylabel( 'Latency (s)' )
        ax.set_xlabel( 'Input Parameter: Number of Iterations')
        ax.set_xlim(0)
        ax.set_ylim(0)
        self.fig.suptitle( title )

if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser( description="Plot MC and Exec Time of functions with changing input size" )
    argparser.add_argument( "--mc_a_dir", '-d', help="Log Directory for exp with mc_a results", required=True, type=str)                
    args = argparser.parse_args()
    
    p_ext = '.pickle'
    
    mc_a_dir = args.mc_a_dir

    mc_sys = mc_a_dir+'/dfs/analysis/mc_sys'+p_ext
    exec_time = mc_a_dir+'/dfs/analysis/exec_time'+p_ext
    
    df_mc_sys = load_pickle( mc_sys )
    df_exec_time  = load_pickle( exec_time  )

    order = reversed(df_mc_sys.columns)
    df_mc_sys = df_mc_sys[order]
    indexes = reversed(df_exec_time.index)
    df_exec_time = df_exec_time.loc[indexes]

    plot = MCBarPlot( 1, 1, 5, 3, mc_a_dir + '/dfs/plots/standalone/comparison_input_sizes_mc_sys' )
    plot.plot_init()
    plot.plot_draw( df_mc_sys, '' )
    plot.plot_close()

    plot = ExecBarPlot( 1, 1, 5, 3, mc_a_dir + '/dfs/plots/standalone/comparison_input_sizes_exec_time' )
    plot.plot_init()
    plot.plot_draw( df_exec_time, '' )
    plot.plot_close()

