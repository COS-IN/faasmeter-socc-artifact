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
        matplotlib.rcParams.update({'font.size': 10})
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


    def plot_draw( self, 
                    desktop_errors, 
                    desktop_nstd, 
                    server_errors, 
                    server_nstd, 
                    title, 
                    ylabel, 
                    xlabel 
                 ):

        ax = self.axs

        d_color = 'tab:red'
        s_color = 'tab:blue'
        
        print( desktop_errors )
        print( desktop_nstd )

        ax.scatter( desktop_nstd, desktop_errors, label='Desktop', marker='s', color=d_color )
        ax.scatter( server_nstd, server_errors, label='Server', marker='o', color=s_color )

        def get_line( v_x, error_x ):
            fit = list(zip(v_x, error_x))
            fit = sorted(fit, key=lambda x: x[0])
            # print(v_d, error_d)
            # print(v_s, error_s)
            # print(fit)
            xs = [p[0] for p in fit]
            ys = [p[1] for p in fit]
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            #xs = np.arange(0, max(xs), 0.1)
            xs = np.arange(0, 0.8, 0.1)
            ys = p(xs)
            return xs,ys
        
        xs_d, ys_d = get_line( desktop_nstd, desktop_errors )
        xs_s, ys_s = get_line( server_nstd, server_errors )

        ax.plot(xs_d, ys_d, color=d_color, linestyle='dotted')
        ax.plot(xs_s, ys_s, color=s_color, linestyle='dotted')

        self.fig.legend(bbox_to_anchor=(0.88, 0.90), ncol=3)
        ax.set_ylabel( ylabel )
        ax.set_xlabel( xlabel )

        ax.set_xlim(0,max(np.concatenate([desktop_nstd , server_nstd]))+0.1)
        ax.set_ylim(0,max(np.concatenate([desktop_errors , server_errors]))+5)

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
    argparser.add_argument( "--dirs_desktop", help="Log Directories for exps to use", required=True, type=str, nargs='+')                
    argparser.add_argument( "--dirs_server", help="Log Directories for exps to use", required=True, type=str, nargs='+')                
    argparser.add_argument( "--save_plot", help="Log Directories for exps to use", required=True, type=str)                
    args = argparser.parse_args()

    p_ext = '.pickle'

    plot_dir = args.save_plot
    
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
    
    desktop_dirs = args.dirs_desktop 
    server_dirs = args.dirs_server 

    do_for_f23 = True 

    def standarize_index( df ):
        d_idx = [ function_name_to_paper(f) for f in df.index ]
        df.index = d_idx
    def get_df_list( dirs ):
        d_exects = []
        d_errors = []
        d_exec_nstd = []

        for d in dirs:
            d_exect = load_df( d, '/dfs/analysis/exec_time' )
            standarize_index( d_exect )
            if do_for_f23:
                d_error = load_df( d, '/dfs/analysis/errors_mean_sys_f23_n' )
            else:
                d_error = load_df( d, '/dfs/analysis/errors_mean_sys_full_n' )
            standarize_index( d_error )
            nstd = d_exect['exec_time']['std'] / d_exect['exec_time']['mean']
            #nstd = d_exect['exec_time']['std'] 
            #nstd = d_exect['exec_time']['mean'] 

            d_exec_nstd.append( nstd )
            d_exects.append( d_exect )
            d_errors.append( d_error )
        d_exec_nstd = np.concatenate( d_exec_nstd )
        d_exects = np.concatenate( d_exects )
        d_errors = np.concatenate( d_errors )
        return d_exects, d_errors, d_exec_nstd
    
    desktop_execs, desktop_errors, desktop_nstd  = get_df_list( desktop_dirs )
    server_execs, server_errors, server_nstd  = get_df_list( server_dirs )
    
    def sort_by_error(derror, dother):
        dother = list(dother)
        derror = list(derror)
        def _sort_by_error(element):
            idx = dother.index(element)
            return derror[idx]
        return sorted(dother, key=_sort_by_error)
    
    desktop_nstd = sort_by_error(desktop_errors, desktop_nstd)
    desktop_errors = sorted( desktop_errors ) 

    server_nstd = sort_by_error(server_errors, server_nstd)
    server_errors = sorted( server_errors ) 
    
    #if False:
    if True:
        threshold = 100.0
        idx_e = next(x[0] for x in enumerate(desktop_errors) if x[1] > threshold) 
        desktop_errors = desktop_errors[0:idx_e]
        desktop_nstd = desktop_nstd[0:idx_e]

        idx_e = next(x[0] for x in enumerate(server_errors) if x[1] > threshold) 
        server_errors = server_errors[0:idx_e]
        server_nstd = server_nstd[0:idx_e]

    # print( desktop_execs )
    
    if do_for_f23:
        plot = ErrorScatterPlot( 1, 1, 3, 3, plot_dir + '/dfs/plots/standalone/error_scatter_plot_all' )
    else:
        plot = ErrorScatterPlot( 1, 1, 3, 3, plot_dir + '/dfs/plots/standalone/error_scatter_plot_all_full' )
    plot.plot_init()
    plot.plot_draw( 
                    desktop_errors, 
                    desktop_nstd, 
                    server_errors, 
                    server_nstd, 
                    '', 
                    r'% Individual-Difference', r'$\sigma( T )\ /\ E[T]$' 
                )

    #plot.plot_draw( funcs_p, e_d, e_s, v_d, v_s, '', r'$\%(error)$', r'$\sigma( T )$' )
    plot.plot_close()

