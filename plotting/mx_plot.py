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

    def plot_draw(self, cols, mcs_cpu, jcpu_mean, jcpu_std, title  ):

        w = 0.4
        offset = w + 0.05
        count = 1
        ax = self.axs

        ax.bar( cols, mcs_cpu, width=w )
        ax.bar( ['f22'], jcpu_mean, width=w )
        ax.errorbar( ['f22'], jcpu_mean, jcpu_std )
        # ax.plot( funcs_size, container, marker='o' )
        
        self.fig.legend(bbox_to_anchor=(1.01, 1.0), ncol=1)
        ax.set_ylabel( 'Energy per invocation (J)' )
        ax.set_xlabel( 'Factor of invocations in (Mca - Mcx) 0.0 means no invocation ')
        ax.set_xlim(0)
        ax.set_ylim(0)
        # plt.xticks(rotation='vertical')
        self.fig.suptitle( title )

if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser( description="Plot MCx and f23, 22 J per invoke" )
    argparser.add_argument( "--mc_a_dir", '-d', help="Log Directory for exp with mc_a results", required=True, type=str)                
    args = argparser.parse_args()
    
    p_ext = '.pickle'
    
    mc_a_dir = args.mc_a_dir

    funcs_analyzed = mc_a_dir+'/dfs/analysis/mcx_funcs_analyzed'+p_ext
    df_funcs_analyzed = load_pickle( funcs_analyzed )

    mcx_sys = mc_a_dir+'/dfs/analysis/mcx_sys_delta_e_per_invks'+p_ext
    df_mcx_sys = load_pickle( mcx_sys )

    mcx_cpu = mc_a_dir+'/dfs/analysis/mcx_cpu_delta_e_per_invks'+p_ext
    df_mcx_cpu = load_pickle( mcx_cpu )

    # F23
    j_per_invk_cpu = mc_a_dir+'/dfs/analysis/j_per_invk_cpu'+p_ext
    df_j_per_invk_cpu = load_pickle( j_per_invk_cpu )

    j_per_invk_shared = mc_a_dir+'/dfs/analysis/j_per_invk_shared'+p_ext
    df_j_per_invk_shared = load_pickle( j_per_invk_shared )

    j_per_invk_full = mc_a_dir+'/dfs/analysis/j_per_invk_full'+p_ext
    df_j_per_invk_full = load_pickle( j_per_invk_full )

    fanalyzed = df_funcs_analyzed[0][0]
    mcs_sys = list(df_mcx_sys.to_numpy()[0])
    mcs_cpu = list(df_mcx_cpu.to_numpy()[0])
    cols = [ str(c.split('_')[2]+'.'+c.split('_')[3]) for c in df_mcx_sys.columns ]
    
    def get_js( df ):
        s = df[fanalyzed]
        return s.mean(), s.std()
    jcpu_mean, jcpu_std = get_js( df_j_per_invk_cpu ) 
    jfull_mean, jfull_std = get_js( df_j_per_invk_full ) 

    print( cols )
    print( mcs_cpu )
    print(jcpu_mean)
    print(jcpu_std)
    print( mcs_sys )
    print(jfull_mean)
    print(jfull_std)

    plot = MCBarPlot( 1, 1, 5, 3, mc_a_dir + '/dfs/plots/standalone/mcx_exp_cpu' )
    plot.plot_init()
    plot.plot_draw( cols, mcs_cpu, jcpu_mean, jcpu_std, fanalyzed + ' cpu'  )
    plot.plot_close()

    plot = MCBarPlot( 1, 1, 5, 3, mc_a_dir + '/dfs/plots/standalone/mcx_exp_sys' )
    plot.plot_init()
    plot.plot_draw( cols, mcs_sys, jfull_mean, jfull_std, fanalyzed + ' sys'  )
    plot.plot_close()

    exit(0)

    plot = ExecBarPlot( 1, 1, 5, 3, mc_a_dir + '/dfs/plots/standalone/comparison_input_sizes_exec_time' )
    plot.plot_init()
    plot.plot_draw( df_exec_time, '' )
    plot.plot_close()

