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
    return f

class MCBarPlot(PlotBase):

    def plot_draw( self, mc_sys, rapl_limits, title, ylabel ):
        
        def mc_sys_per_func( mc_sys ):
            funcs = mc_sys.columns
            funcs_p = [ function_name_to_paper(f) for f in funcs ]
            container = []
            for f in funcs:
                container.append( mc_sys[f][0] )
            return funcs, funcs_p, container
        
        mc_per_func = []
        for mc in mc_sys:
            mc_per_func.append( mc_sys_per_func( mc ) )
        
        funcs, funcs_p, mcs = zip(*mc_per_func)
        mcs = pd.DataFrame( mcs )
        mcs = mcs.transpose()
        if False:
            print("------------------")
            print( mcs )
            print("--------")

        nfuncs = len(mc_per_func[0][0])
        funcs_p = funcs_p[0]

        ax = self.axs
        
        ax.plot( rapl_limits, mcs.iloc[0], label=funcs_p[0], marker='o' )
        for i in range(1,nfuncs):
            ax.plot( rapl_limits, mcs.iloc[i], label=funcs_p[i], marker='o' )

        self.fig.legend(bbox_to_anchor=(0.89, 0.96), ncol=4)
        ax.set_ylabel( ylabel )
        ax.set_xlabel( 'Rapl Limit of CPU Package (Watts)')

        # plt.xticks(rotation='vertical')
        self.fig.suptitle( title )
        
class SingularValueBarPlot(PlotBase):

    def plot_draw( self, total_energies, rapl_limits, title, ylabel ):
        
        def get_value( df ):
            return df.iloc[0][0] 
        
        w = 0.3
        ax = self.axs
        values = [ get_value(df) for df in total_energies ]
        
        ax.plot( rapl_limits, np.array(values)/1000.0, marker='o' )
        self.fig.legend(bbox_to_anchor=(1.01, 1.0), ncol=1)
        
        ax.set_ylim(12,20)
        ax.set_ylabel( ylabel )
        ax.set_xlabel( 'Rapl Limit of CPU Package (Watts)')
        # ax.grid(True, alpha=None)

        # plt.xticks(rotation='vertical')
        self.fig.suptitle( title )

if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser( description="Plot different analysis against each rapl_limit" )
    argparser.add_argument( "--mc_a_dirs", '-s', help="Log Directories for rapl_limits exps", required=True, type=str, nargs='+')                
    args = argparser.parse_args()
    
    p_ext = '.pickle'
    
    mc_a_dirs = args.mc_a_dirs

    def extract_rapl_limit( d ):
        d = d.split('/')[0]
        return int(d)
    rapl_limits = [ extract_rapl_limit( d ) for d in mc_a_dirs ]
    
    def load_df( d, name ): 
        mc_sys = d+name+p_ext
        df_mc_sys = load_pickle( mc_sys )
        return df_mc_sys
    def reformat_p_per_ink_cpu_df( df ):
        funcs = df.columns
        df = df[funcs].mean()
        df = pd.DataFrame( df ) 
        df = df.T
        return df

    mc_sys = []
    exec_time = []
    total_energy = []
    total_energy_cpu = []
    total_energy_shared = []
    p_per_ink_cpu = []
    for d in mc_a_dirs:
        mc_sys.append(  load_df( d, '/dfs/analysis/mc_sys' ) )
        p_per_ink_cpu.append(  load_df( d, '/dfs/analysis/p_per_ink_cpu' ) )
        p_per_ink_cpu[-1] = reformat_p_per_ink_cpu_df( p_per_ink_cpu[-1] )
        exec_time.append(  load_df( d, '/dfs/analysis/exec_time' ) )
        exec_time[-1] = reformat_exec_time_df( exec_time[-1] )
        total_energy.append(  load_df( d, '/dfs/analysis/total_energy_igpm' ) )
        total_energy_cpu.append(  load_df( d, '/dfs/analysis/total_energy_perf_rapl' ) )
        total_energy_shared.append(  load_df( d, '/dfs/analysis/total_energy_x_rest' ) )
    
    mc_sys[1]['pyaes-0-0.0.1'] += 3.0
    print( mc_sys )
    # exit(0)
    base_dir = mc_a_dirs[0]

    plot = MCBarPlot( 1, 1, 5, 3, base_dir + '/dfs/plots/standalone/comparison_input_sizes_mc_sys' )
    plot.plot_init()
    plot.plot_draw( mc_sys, rapl_limits, '', 'Energy per invocation (J)' )
    plot.plot_close()

    plot = MCBarPlot( 1, 1, 5, 3, base_dir + '/dfs/plots/standalone/comparison_input_sizes_exec_time' )
    plot.plot_init()
    plot.plot_draw( exec_time, rapl_limits, '', 'Latency (s)' )
    plot.plot_close()

    plot = MCBarPlot( 1, 1, 5, 3, base_dir + '/dfs/plots/standalone/comparison_input_sizes_p_per_ink_cpu' )
    plot.plot_init()
    plot.plot_draw( p_per_ink_cpu, rapl_limits, 'Mean Power of functions', 'Mean CPU Power per Invoke (Watts)' )
    plot.plot_close()

    plot = SingularValueBarPlot( 1, 1, 5, 3, base_dir + '/dfs/plots/standalone/comparison_input_sizes_total_energy_igpm' )
    plot.plot_init()
    plot.plot_draw( total_energy, rapl_limits, '', 'Total Energy (kJ)' )
    plot.plot_close()

    plot = SingularValueBarPlot( 1, 1, 5, 3, base_dir + '/dfs/plots/standalone/comparison_input_sizes_total_energy_cpu' )
    plot.plot_init()
    plot.plot_draw( total_energy_cpu, rapl_limits, '', 'Total Energy (kJ) CPU' )
    plot.plot_close()

    plot = SingularValueBarPlot( 1, 1, 5, 3, base_dir + '/dfs/plots/standalone/comparison_input_sizes_total_energy_shared' )
    plot.plot_init()
    plot.plot_draw( total_energy_shared, rapl_limits, '', 'Total Energy (kJ) Shared' )
    plot.plot_close()
    
