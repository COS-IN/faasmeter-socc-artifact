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

        matplotlib.rcParams['figure.subplot.left'] = 0.12  # the left side of the subplots of the figure
        matplotlib.rcParams['figure.subplot.right'] = 0.95    # the right side of the subplots of the figure
        matplotlib.rcParams['figure.subplot.bottom'] = 0.15   # the bottom of the subplots of the figure
        matplotlib.rcParams['figure.subplot.top'] = 0.8      # the top of the subplots of the figure
        matplotlib.rcParams['figure.subplot.wspace'] = 0.25   # the amount of width reserved for blank space between subplots
        matplotlib.rcParams['figure.subplot.hspace'] = 0.2   # the amount of height reserved for white space between subplots
        # matplotlib.rcParams['text.usetex'] = True

    def plot_draw( self, errors_f22_norm, errors_f22_cosine, errors_f23_norm, errors_f23_cosine, subtitles, ylabel ):
        
        ax = self.axs[0]
        
        fsize = 3 
        w = 0.15
        offset = w + 0.05
        count = 1
        def spit_label( l ):
            return l

        handles = {} 

        h = ax.bar( subtitles, errors_f22_cosine, width=w, label=spit_label('Full Dissaggregation') )
        ax.bar_label(h, fmt='%.1f' , fontsize=fsize)
        handles['f22_cos'] =  h 

        x = ax.get_xticks()
        x = np.array(x, dtype=np.float64)

        x += offset 
        h = ax.bar( x, errors_f23_cosine, width=w, label=spit_label('Combined Dissaggregation') )
        ax.bar_label(h, fmt='%.1f' , fontsize=fsize)
        handles['f23_cos'] =  h 

        ax.set_ylabel('Cosine Similarity')
       
        ax = self.axs[1]

        fsize = 3 
        w = 0.15
        offset = w + 0.05
        count = 1

        handles = {} 

        h = ax.bar( subtitles, errors_f22_norm, width=w )
        ax.bar_label(h, fmt='%.1f' , fontsize=fsize)
        handles['f22_norm'] =  h 

        x = ax.get_xticks()
        x = np.array(x, dtype=np.float64)

        x += offset 
        h = ax.bar( x, errors_f23_norm, width=w )
        ax.bar_label(h, fmt='%.1f' , fontsize=fsize)
        handles['f23_norm'] =  h 

        ax.set_ylabel('L2Norm')

        self.fig.legend(bbox_to_anchor=(0.95, 1.0), ncol=4)

        # plt.xticks(rotation='vertical')
        # self.fig.suptitle( title )
        
if __name__ == '__main__':

    import argparse

    argparser = argparse.ArgumentParser( description="Combined Jetson, Desktop, Server results" )
    argparser.add_argument( "--mc_a_dirs", '-s', help="Log Directories for mc_a exps of the three platforms", required=True, type=str, nargs='+')                
    args = argparser.parse_args()

    function_set = [
                    'dd-0-0.0.1', 
                    'image_processing-0-0.0.1', 
                    'pyaes-0-0.0.1', 
                    'cnn_image_classification-0-0.0.1'
    ]
    
    print("#"*20)
    print("# Plotting is limited: \n\t{}".format(function_set))
    
    p_ext = '.pickle'
    
    mc_a_dirs = args.mc_a_dirs

    def extract_rapl_limit( d ):
        d = d.split('/')[0]
        return d
    rapl_limits = [ extract_rapl_limit( d ) for d in mc_a_dirs ]
    
    def load_df( d, name ): 
        print(d+name)
        mc_sys = d+name+p_ext
        df_mc_sys = load_pickle( mc_sys )
        return float(df_mc_sys[0][0])

    errors_f22_cosine = []
    errors_f23_cosine = []
    errors_f22_norm = []
    errors_f23_norm = []

    for i,d in enumerate(mc_a_dirs):
            
        def try_fill( ls, name ):
            try: 
                ls.append( load_df(d, '/dfs/analysis/'+name) )
            except FileNotFoundError:
                ls.append( 0.0 )
  
        try_fill( errors_f23_cosine, 'errors_distance_sys_cosine' )
        try_fill( errors_f22_cosine, 'errors_distance_sys_full_cosine' )

        try_fill( errors_f23_norm, 'errors_distance_sys_norm' )
        try_fill( errors_f22_norm, 'errors_distance_sys_full_norm' )
    
    print( errors_f22_norm )
    print( errors_f22_cosine )
    print( errors_f23_norm )
    print( errors_f23_cosine )

    base_dir = mc_a_dirs[0]

    plot = MCBarPlot( 1, 2, 5, 2.0, base_dir + '/dfs/plots/standalone/fig_9_errors' )
    plot.plot_init()
    plot.plot_draw( 
                    errors_f22_norm,
                    errors_f22_cosine,
                    errors_f23_norm,
                    errors_f23_cosine,
                    # ['Desktop', 'Server'], 
                    ['Desktop', 'Server', 'Jetson'], 
                    'J per Invoke (Watts)' 
                  )
    plot.plot_close()

