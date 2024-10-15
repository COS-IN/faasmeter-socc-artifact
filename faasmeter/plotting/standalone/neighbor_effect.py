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

class NeighborBarPlot(PlotBase):

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
        matplotlib.rcParams['figure.subplot.right'] = 0.75    # the right side of the subplots of the figure
        matplotlib.rcParams['figure.subplot.bottom'] = 0.10   # the bottom of the subplots of the figure
        matplotlib.rcParams['figure.subplot.top'] = 1.0      # the top of the subplots of the figure
        matplotlib.rcParams['figure.subplot.wspace'] = 0.25   # the amount of width reserved for blank space between subplots
        matplotlib.rcParams['figure.subplot.hspace'] = 0.2   # the amount of height reserved for white space between subplots

        matplotlib.rcParams['legend.frameon'] = False   
        matplotlib.rcParams['legend.fancybox'] = False   

        matplotlib.rcParams['legend.handletextpad'] = 0.1   
        matplotlib.rcParams['legend.borderpad'] = 0.1   
        matplotlib.rcParams['legend.borderaxespad'] = 0.1   



    def plot_draw( self, funcs_p, m_wdd, m_wml, gt_wdd, gt_wml,  title, ylabel, xlabel ):
        
        colors = ['0.5', 'black', 'red', 'saddlebrown']
        ax = self.axs

        plot_data = []
        for i, f in enumerate(funcs_p):
          plot_data.append([])
          if m_wdd[i] != 0.0:
            plot_data[i].append(m_wdd[i])
          if m_wml[i] != 0.0:
            plot_data[i].append(m_wml[i])
          if gt_wdd[i] != 0.0:
            plot_data[i].append(gt_wdd[i])
          if gt_wml[i] != 0.0:
            plot_data[i].append(gt_wml[i])

        # print(plot_data)
        w = 0.18
        space = 0.02
        offset = w + space
        start = 0
        tick_positions = []
        for i, data in enumerate(plot_data):
          tick_positions.append(start)
          if len(data) == 4:
            pts = [start-(w*1.5)-2*space, start-(w/2)-space, start+(w/2)+space, start+(w*1.5+2*space)]
            if len(plot_data[i+1]) == 4:
              start += 4*w + w
            else:
              start += 3*w + w
          if len(data) == 2:
            pts = [start-w/2 - space, start+w/2 + space]
            start += 2*w + w
          # print(funcs_p[i], pts)
          if funcs_p[i] != "ml_train":
            for j in range(len(data)):
                ax.bar(pts[j], data[j], width=w, color=colors[j] )
          else:
            ax.bar(pts[0], data[0], width=w, color=colors[2] )
            ax.bar(pts[1], data[1], width=w, color=colors[3] )
        
        ax.scatter( -1, 0, marker='s', label='Measured', color='white' ) 
        ax.scatter( -1, 0, marker='s', label='dd', color=colors[0] ) 
        ax.scatter( -1, 0, marker='s', label='ml_train', color=colors[2]) 
        ax.scatter( -1, 0, marker='s', label='                  ', color='white' ) 
        ax.scatter( -1, 0, marker='s', label='Ground Truth', color='white' ) 
        ax.scatter( -1, 0, marker='s', label='dd', color=colors[1] ) 
        ax.scatter( -1, 0, marker='s', label='ml_train', color=colors[3] ) 
        self.fig.legend(bbox_to_anchor=(0.96, 0.80), ncol=1)

        ax.set_ylabel( ylabel )
        ax.set_xlabel( xlabel )
        # print(tick_positions)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(funcs_p)

        ax.set_xlim(-0.5 )
        ax.set_ylim(0)

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
        # print(mc_sys)
        df_mc_sys = load_pickle( mc_sys )
        return df_mc_sys
    def reformat_p_per_ink_cpu_df( df ):
        funcs = df.columns
        df = df[funcs].mean()
        df = pd.DataFrame( df ) 
        df = df.T
        return df
    
    wdd_dir = mc_a_dirs[0]
    wml_dir = mc_a_dirs[1]

    def get_dfs( _dir ):
        _jcpu = load_df( _dir, '/dfs/analysis/j_per_invk_cpu' )
        _jshared = load_df( _dir, '/dfs/analysis/j_per_invk_shared' )
        _f23 = _jcpu + _jshared
        _f23 = _f23.mean()
        _f23 = pd.DataFrame( _f23 ).T
        _mc_sys = load_df( _dir, '/dfs/analysis/mc_sys' )
        return _f23, _mc_sys

    wdd_f23, wdd_mc_sys = get_dfs( wdd_dir )
    wml_f23, wml_mc_sys = get_dfs( wml_dir )

    # data formatting for plotting
    funcs_wdd = list(wdd_mc_sys.columns)
    funcs_wml = list(wml_mc_sys.columns)

    funcs_common = set(funcs_wdd).intersection( set(funcs_wml) )
    funcs_common = list(funcs_common)

    funcs_wdd = list(set(funcs_wdd).difference( set(funcs_common) ))
    funcs_wml = list(set(funcs_wml).difference( set(funcs_common) ))
    
    funcs_all = funcs_common + funcs_wdd + funcs_wml
    funcs_p = [ function_name_to_paper(f) for f in funcs_all ]

    print( funcs_p )
    
    m_wdd = wdd_f23[funcs_common + funcs_wdd]
    m_wdd[funcs_wml] = 0.0
    gt_wdd = wdd_mc_sys[funcs_common + funcs_wdd]
    gt_wdd[funcs_wml] = 0.0

    m_wml = wml_f23[funcs_common]
    m_wml[funcs_wdd] = 0.0
    m_wml[funcs_wml] = wml_f23[funcs_wml]

    gt_wml = wml_mc_sys[funcs_common]
    gt_wml[funcs_wdd] = 0.0
    gt_wml[funcs_wml] = wml_mc_sys[funcs_wml]

    plot = NeighborBarPlot( 1, 1, 8, 2.7, wdd_dir + '/dfs/plots/standalone/neighboreffect_plot' )
    plot.plot_init()
    plot.plot_draw( 
                    funcs_p, 
                    m_wdd.to_numpy()[0], 
                    m_wml.to_numpy()[0], 
                    gt_wdd.to_numpy()[0], 
                    gt_wml.to_numpy()[0], 
                    '', 
                    r'Energy per invocation (J)', 
                    r'' 
                  )
    plot.plot_close()

