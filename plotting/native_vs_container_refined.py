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

class MCBarPlot(PlotBase):

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
        matplotlib.rcParams['figure.subplot.top'] = 0.85      # the top of the subplots of the figure
        matplotlib.rcParams['figure.subplot.wspace'] = 0.25   # the amount of width reserved for blank space between subplots
        matplotlib.rcParams['figure.subplot.hspace'] = 0.2   # the amount of height reserved for white space between subplots

        matplotlib.rcParams['legend.frameon'] = False   
        matplotlib.rcParams['legend.fancybox'] = False   

        matplotlib.rcParams['legend.handletextpad'] = 0.1   
        matplotlib.rcParams['legend.borderpad'] = 0.1   
        matplotlib.rcParams['legend.borderaxespad'] = 0.1   

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
        
        if False:
            end = -2
            native_o = native[end:]
            container_o = container[end:]
            funcs_p_o = funcs_p[end:]

            native = native[0:end]
            container = container[0:end]
            funcs_p = funcs_p[0:end]

        ax.bar( funcs_p, native, width=w, label='Baremetal' )

        x = ax.get_xticks()
        x = np.array(x, dtype=np.float64)
        x += offset * count
        ax.bar( x, container, width=w, label='FaaS' )

        axins = ax.inset_axes([0.13, 0.3, 0.2, 0.47])

        axins.bar( funcs_p, native, width=w, label='Baremetal' )

        x = axins.get_xticks()
        x = np.array(x, dtype=np.float64)
        x += offset * count
        axins.bar( x, container, width=w, label='FaaS' )

        axins.set_xlim(-offset/1.5, w + offset/1.5 )
        axins.set_ylim( 0, 0.5 )

        ax.indicate_inset_zoom(axins, edgecolor="black") 

        self.fig.legend(bbox_to_anchor=(0.68, 0.92), ncol=2)
        ax.set_ylabel( r'Energy per Invocation (J)' )
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

        ax.bar( funcs_p, native, width=w, label='Baremetal' )

        x = ax.get_xticks()
        x = np.array(x, dtype=np.float64)
        x += offset * count
        ax.bar( x, container, width=w, label='FaaS' )

        self.fig.legend(bbox_to_anchor=(0.68, 0.92), ncol=2)
        ax.set_ylabel( 'Latency (s)' )
        self.fig.suptitle( title )


if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser( description="Plot Native vs Container comparison" )
    argparser.add_argument( "--cont_dirs", '-c', help="Log Directories for single experiment results", required=True, type=str, nargs='+')                
    argparser.add_argument( "--native_dirs", '-n', help="Log Directories for native results", required=True, type=str, nargs='+')                
    args = argparser.parse_args()
    
    p_ext = '.pickle'
    
    def get_c_n_dfs( cd, nd ):
        c_exec_time = cd+'/dfs/analysis/exec_time'+p_ext
        c_mc_sys = cd+'/dfs/analysis/mc_sys'+p_ext
        n_e_per_invk = nd+'/dfs/e_per_invk'+p_ext
        
        df_c_exec_time  = load_pickle( c_exec_time  )
        df_c_mc_sys = load_pickle( c_mc_sys )
        df_n_e_per_invk  = load_pickle( n_e_per_invk  )
        return df_c_exec_time, df_c_mc_sys, df_n_e_per_invk 

    exec_time_c, mc_c, native_stuff = get_c_n_dfs( args.cont_dirs[0], args.native_dirs[0] )
    exec_time_c_1, mc_c_1, native_stuff_1 = get_c_n_dfs( args.cont_dirs[1], args.native_dirs[1] )
    
    f = 'pyaes-0-0.0.1' 
    exec_time_c = exec_time_c.append( exec_time_c_1.loc[f] )
    mc_c[f] = mc_c_1[f]
    f = 'pyaes'
    native_stuff = native_stuff.append( native_stuff_1.loc[f] )
    
    def get_std_names( iterable ):
        names = []
        for i,f in enumerate(iterable):
            names.append( function_name_to_paper( f )  )
        return names
    
    exec_time_c.index = get_std_names( exec_time_c.index )
    mc_c.columns = get_std_names( mc_c.columns )
    order = [ 
                "json", 
                "AES_small",
                "AES", 
            ]
    mc_c = mc_c[order]
    native_stuff.index = get_std_names( native_stuff.index )

    print('-------------------------')
    print( exec_time_c )
    print( mc_c )
    print( native_stuff )
    
    nd = args.native_dirs[0]
    plot = MCBarPlot( 1, 1, 4, 3, nd + '/comparison_to_cont_mc' )
    plot.plot_init()
    plot.plot_draw( mc_c, native_stuff, '' )
    plot.plot_close()

    plot = ExecBarPlot( 1, 1, 4, 3, nd + '/comparison_to_cont_exec_time' )
    plot.plot_init()
    plot.plot_draw( exec_time_c, native_stuff, '' )
    plot.plot_close()

