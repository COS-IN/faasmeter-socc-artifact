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

class MCBarPlot(PlotBase):

    def plot_draw( self, c_mc_sys, n_all, title ):
        funcs = c_mc_sys.columns
        funcs_p = [ function_name_to_paper(f) for f in funcs ]

        j_per_invk = n_all['e_per_invk']

        container = []
        native = []

        for f in funcs:
            fn = f.split('-')[0]
            container.append( c_mc_sys[f][0] )
            native.append( j_per_invk.loc[fn] )

        w = 0.4
        offset = w + 0.05
        count = 1
        ax = self.axs

        ax.bar( funcs_p, native, width=w, label='Native' )

        x = ax.get_xticks()
        x = np.array(x, dtype=np.float64)
        x += offset * count
        ax.bar( x, container, width=w, label='Containerized' )

        plt.axvspan(x[0] -(offset*count + w/2 +offset/5) , x[1] + w/2 + offset/5, color='red', alpha=0.2)
        plt.text( 0.0, -7.0, "Disk Intensive" )

        self.fig.legend(bbox_to_anchor=(1.01, 1.0), ncol=1)
        ax.set_ylabel( 'J per Invoke (Watts)' )
        self.fig.suptitle( title )


class ExecBarPlot(PlotBase):

    def plot_draw( self, c_exec_time, n_all, title ):
        funcs = c_exec_time.index
        funcs_p = [ function_name_to_paper(f) for f in funcs ]

        e2e = n_all['e2e']

        container = []
        native = []

        for f in funcs:
            fn = f.split('-')[0]
            container.append( c_exec_time.loc[f]['exec_time']['mean'] )
            native.append( e2e.loc[fn] )
        
        w = 0.4
        offset = w + 0.05
        count = 1
        ax = self.axs

        ax.bar( funcs_p, native, width=w, label='Native' )

        x = ax.get_xticks()
        x = np.array(x, dtype=np.float64)
        x += offset * count
        ax.bar( x, container, width=w, label='Containerized' )

        plt.axvspan(x[0] -(offset*count + w/2 +offset/5) , x[1] + w/2 + offset/5, color='red', alpha=0.2)
        plt.text( 0.0, -2.0, "Disk Intensive" )

        self.fig.legend(bbox_to_anchor=(1.01, 1.0), ncol=1)
        ax.set_ylabel( 'End to End Time (seconds)' )
        self.fig.suptitle( title )


if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser( description="Plot Native vs Container comparison" )
    argparser.add_argument( "--cont_dir", '-c', help="Log Directory for single experiment results", required=True, type=str)                
    argparser.add_argument( "--native_dir", '-n', help="Log Directory for native results", required=True, type=str)                
    args = argparser.parse_args()
    
    p_ext = '.pickle'
    
    cd = args.cont_dir
    nd = args.native_dir

    c_exec_time = cd+'/dfs/analysis/exec_time'+p_ext
    c_mc_sys = cd+'/dfs/analysis/mc_sys'+p_ext
    n_e_per_invk = nd+'/dfs/e_per_invk'+p_ext
    
    df_c_exec_time  = load_pickle( c_exec_time  )
    df_c_mc_sys = load_pickle( c_mc_sys )
    df_n_e_per_invk  = load_pickle( n_e_per_invk  )
    
    plot = MCBarPlot( 1, 1, 5, 3, nd + '/comparison_to_cont_mc' )
    plot.plot_init()
    plot.plot_draw( df_c_mc_sys, df_n_e_per_invk, 'Native vs Containerized' )
    plot.plot_close()

    plot = ExecBarPlot( 1, 1, 5, 3, nd + '/comparison_to_cont_exec_time' )
    plot.plot_init()
    plot.plot_draw( df_c_exec_time, df_n_e_per_invk, 'Native vs Containerized' )
    plot.plot_close()

