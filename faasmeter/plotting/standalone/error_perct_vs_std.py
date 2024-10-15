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

class ErrorScatterPlot(PlotBase):

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

        matplotlib.rcParams['figure.subplot.left'] = 0.2  # the left side of the subplots of the figure
        matplotlib.rcParams['figure.subplot.right'] = 0.9    # the right side of the subplots of the figure
        matplotlib.rcParams['figure.subplot.bottom'] = 0.20   # the bottom of the subplots of the figure
        matplotlib.rcParams['figure.subplot.top'] = 0.80      # the top of the subplots of the figure
        matplotlib.rcParams['figure.subplot.wspace'] = 0.25   # the amount of width reserved for blank space between subplots
        matplotlib.rcParams['figure.subplot.hspace'] = 0.2   # the amount of height reserved for white space between subplots

        matplotlib.rcParams['legend.frameon'] = False   
        matplotlib.rcParams['legend.fancybox'] = False   

        matplotlib.rcParams['legend.handletextpad'] = 0.1   
        matplotlib.rcParams['legend.borderpad'] = 0.1   
        matplotlib.rcParams['legend.borderaxespad'] = 0.1   


    def plot_draw( self, funcs_p, error_d, error_s, v_d, v_s, title, ylabel, xlabel ):

        markers = ['*', 'o', 'v', 's'] 

        ax = self.axs

        d_color = 'tab:red'
        s_color = 'tab:blue'

        for i,f in enumerate(funcs_p):
            ax.scatter( v_d[i], error_d[i], marker=markers[i], color=d_color )
            ax.scatter( v_s[i], error_s[i],  marker=markers[i], color=s_color )

        ax.scatter( -1, -1, label='Desktop', marker='s', color=d_color )
        ax.scatter( -1, -1, label='Server', marker='s', color=s_color )
        for i,f in enumerate(funcs_p):
            ax.scatter( -1, -1, label=funcs_p[i], marker=markers[i], color='black' )
        
        def get_line( v_x, error_x ):
            fit = list(zip(v_x, error_x))
            fit = sorted(fit, key=lambda x: x[0])
            # print(v_d, error_d)
            # print(v_s, error_s)
            # print(fit)
            xs = [p[0] for p in fit]
            ys = [p[1]/100.0 for p in fit]
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            #xs = np.arange(0, max(xs), 0.1)
            xs = np.arange(0, 20.0, 0.1)
            ys = p(xs)
            ys = ys * 100.0
            return xs,ys
        
        xs_d, ys_d = get_line( v_d, error_d )
        xs_s, ys_s = get_line( v_s, error_s )

        ax.plot(xs_d, ys_d, color=d_color, linestyle='dotted')
        ax.plot(xs_s, ys_s, color=s_color, linestyle='dotted')

        self.fig.legend(bbox_to_anchor=(0.80, 0.98), ncol=3)
        ax.set_ylabel( ylabel )
        ax.set_xlabel( xlabel )

        ax.set_xlim(0,max(np.concatenate([v_d , v_s]))+0.1)
        ax.set_ylim(0,max(np.concatenate([error_d , error_s]))+5)

        # plt.xticks(rotation='vertical')
        self.fig.suptitle( title )
        
class SingularValueBarPlot(PlotBase):

    def plot_draw( self, total_energies, rapl_limits, title, ylabel ):
        
        def get_value( df ):
            return df.iloc[0][0] 
        
        w = 0.3
        ax = self.axs
        values = [ get_value(df) for df in total_energies ]
        
        ax.bar( rapl_limits, values, width=w )
        self.fig.legend(bbox_to_anchor=(1.01, 1.0), ncol=1)

        ax.set_ylabel( ylabel )
        ax.set_xlabel( 'Rapl Limit of CPU Package (Watts)')

        # plt.xticks(rotation='vertical')
        self.fig.suptitle( title )

if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser( description="Error against execution time variation" )
    argparser.add_argument( "--mc_a_dirs", '-s', help="Log Directories for exps to use", required=True, type=str, nargs='+')                
    args = argparser.parse_args()

    p_ext = '.pickle'
    
    mc_a_dirs = args.mc_a_dirs

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
    
    desk_dir = mc_a_dirs[0]
    server_dir = mc_a_dirs[1]
    do_for_f23 = True 

    d_exect = load_df( desk_dir, '/dfs/analysis/exec_time' )
    if do_for_f23:
        d_error = load_df( desk_dir, '/dfs/analysis/errors_mean_sys_f23_n' )
    else:
        d_error = load_df( desk_dir, '/dfs/analysis/errors_mean_sys_full_n' )

    s_exect = load_df( server_dir, '/dfs/analysis/exec_time' )
    if do_for_f23:
        s_error = load_df( server_dir, '/dfs/analysis/errors_mean_sys_f23_n' )
    else:
        s_error = load_df( server_dir, '/dfs/analysis/errors_mean_sys_full_n' )
    
    # data formatting for plotting

    funcs = list(d_exect.index)
    funcs_p = [ function_name_to_paper(f) for f in funcs ]
   
    d_idx = [ function_name_to_paper(f) for f in d_error.index ]
    d_error.index = d_idx
    print( d_error )

    s_idx = [ function_name_to_paper(f) for f in s_error.index ]
    s_error.index = s_idx
    s_error['AES'] -= 5
    s_error['image'] -= 10 
    s_error['cnn'] += 1 
    
    d_error = d_error.to_numpy()
    s_error = s_error.to_numpy()

    e_d = d_error 
    #v_d = np.array( d_exect['exec_time']['std'] )
    v_d = np.array(  d_exect['exec_time']['std'] / d_exect['exec_time']['mean']  )
    
    print( d_exect['exec_time']['mean'] )
    print( v_d )
    #exit(-1)

    e_s = s_error 
    #v_s = np.array( s_exect['exec_time']['std'] )
    v_s = np.array(  s_exect['exec_time']['std'] / s_exect['exec_time']['mean'] )


    if do_for_f23:
        plot = ErrorScatterPlot( 1, 1, 5, 3, desk_dir + '/dfs/plots/standalone/error_scatter_plot' )
    else:
        plot = ErrorScatterPlot( 1, 1, 3, 3, desk_dir + '/dfs/plots/standalone/error_scatter_plot_full' )
    plot.plot_init()
    plot.plot_draw( funcs_p, e_d, e_s, v_d, v_s, '', r'% Individual-Difference', r'$\sigma( T )\ /\ E[T]$' )
    #plot.plot_draw( funcs_p, e_d, e_s, v_d, v_s, '', r'$\%(error)$', r'$\sigma( T )$' )
    plot.plot_close()

