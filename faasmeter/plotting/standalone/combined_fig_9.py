import os, sys, path
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

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

import re
def is_gpu( string ):
    r = re.compile('.*gpu.*')
    if re.match(r, string):
        return True
    return False

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
                                sharey=False,
                                gridspec_kw={'width_ratios': [4, 4, 5]}
        )
        
        self.fig = fig
        self.axs = axs

        w = 0.2
        self.bar_w = w
        self.bar_offset = w + 0.03
        self.count = 0

    def init_setrcParams(self):
        matplotlib.rcParams.update({'font.size': 9})
        matplotlib.rcParams['pdf.fonttype'] = 42
        # matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
        matplotlib.rcParams['lines.linewidth'] = 1.5
        matplotlib.rcParams["pdf.use14corefonts"] = True
        matplotlib.rcParams['axes.linewidth'] = 0.5
        matplotlib.rcParams['axes.labelpad'] = 2.0
        matplotlib.rcParams['axes.titlepad'] = 2.0
        matplotlib.rcParams['figure.dpi'] = 200.0

        matplotlib.rcParams['figure.subplot.left'] = 0.10  # the left side of the subplots of the figure
        matplotlib.rcParams['figure.subplot.right'] = 0.95    # the right side of the subplots of the figure
        matplotlib.rcParams['figure.subplot.bottom'] = 0.12   # the bottom of the subplots of the figure
        matplotlib.rcParams['figure.subplot.top'] = 0.67      # the top of the subplots of the figure
        matplotlib.rcParams['figure.subplot.wspace'] = 0.15   # the amount of width reserved for blank space between subplots
        matplotlib.rcParams['figure.subplot.hspace'] = 0.1   # the amount of height reserved for white space between subplots
        # matplotlib.rcParams['text.usetex'] = True

    def plot_draw( self, scaphandre_e_per_invk, mc_sys, mc_cpu, mc_shared, j_per_invk_cpu, j_per_invk_shared, j_per_invk_full, subtitles, ylabel, jcgpu = None  ):
        
        def mc_df_per_func( mc_df ):
            funcs = mc_df.columns
            funcs_p = [ function_name_to_paper(f) for f in funcs ]
            container = []
            for f in funcs:
                container.append( mc_df[f][0] )
            return funcs, funcs_p, container
        def jpi_per_func( jpi, transfomer_func ):
            funcs = jpi.columns
            funcs_p = [ function_name_to_paper(f) for f in funcs ]
            jpi = transfomer_func( jpi, axis=0 ) 
            jpi = pd.DataFrame( jpi ).T
            container = []
            for f in funcs:
                container.append( jpi[f][0] )
            return funcs, funcs_p, container
        def jpi_per_func_mean( jpi ):
            return jpi_per_func( jpi, np.mean )
        def jpi_per_func_std( jpi ):
            return jpi_per_func( jpi, np.std )
       
        def list_to_df( df_list, transformer_func ):
            mcp = []
            for mc in df_list:
                mcp.append( transformer_func( mc ) )
            funcs, funcs_p, mcs = zip(*mcp)
            mcs = pd.DataFrame( mcs )
            mcs = mcs.transpose()
            return funcs, funcs_p, mcs

        funcs, funcs_p, mcp_sys = list_to_df( mc_sys, mc_df_per_func )
        funcs, funcs_p, mcp_cpu = list_to_df( mc_cpu, mc_df_per_func )
        funcs, funcs_p, mcp_shared = list_to_df( mc_shared, mc_df_per_func )
        funcs, funcs_p, sca_df = list_to_df( scaphandre_e_per_invk, mc_df_per_func )
        print( sca_df )

        funcs, funcs_p, jpi_cpu_mean = list_to_df( j_per_invk_cpu, jpi_per_func_mean )
        funcs, funcs_p, jpi_cpu_std = list_to_df( j_per_invk_cpu, jpi_per_func_std )
        funcs, funcs_p, jpi_shared_mean = list_to_df( j_per_invk_shared, jpi_per_func_mean )
        funcs, funcs_p, jpi_shared_std = list_to_df( j_per_invk_shared, jpi_per_func_std )
        funcs, funcs_p, jpi_full_mean = list_to_df( j_per_invk_full, jpi_per_func_mean )
        funcs, funcs_p, jpi_full_std = list_to_df( j_per_invk_full, jpi_per_func_std )

        funcs_ps = funcs_p[0]
        nfuncs = len(funcs_ps)
        
        for i, subt in enumerate(subtitles):

            ax = self.axs[i]
            
            fsize = 5.0 
            w = 0.17
            offset = w + 0.05
            count = 1
            def spit_label( l ):
                if i == 0:
                    return l
                return None
            def set_barlabel( h ):
                labels = [ "%.1f" % v if v > 0.0 else "" for v in h.datavalues]
                ax.bar_label(h, labels=labels, fontsize=fsize, rotation=35.0)

            handles = {} 
            
            mc_sys = mcp_sys.loc[:,i]
            mc_cpu = mcp_cpu.loc[:,i]
            mean_cpu = jpi_cpu_mean.loc[:,i] 
            f23mean = jpi_cpu_mean.loc[:,i] + jpi_shared_mean.loc[:,i]
            f23std = jpi_cpu_std.loc[:,i] + jpi_shared_std.loc[:,i]
            fullmean = jpi_full_mean.loc[:,i] 
            fullstd = jpi_full_std.loc[:,i] 
            sca_values = sca_df.loc[:,i] 

            color_map = plt.get_cmap('Accent')
            color_map = plt.get_cmap('tab20')
            color_map = color_map(list(range(0,20)))

            if i == 2 and jcgpu is not None:
                cgpu_paper, cnn_gpu_mc, cnn_gpu_jpi_mean, cnn_gpu_jpi_std = jcgpu
                funcs_ps.append( cgpu_paper )
                mc_sys = mc_sys.append( cnn_gpu_mc )
                mc_cpu = mc_cpu.append( pd.Series([0.0]) )
                mean_cpu = mean_cpu.append( pd.Series([0.0]) )
                f23mean = f23mean.append( pd.Series([0.0]) )
                f23std = f23std.append( pd.Series([0.0]) )
                fullmean = fullmean.append( pd.Series([cnn_gpu_jpi_mean]) )
                fullstd = fullstd.append( pd.Series([cnn_gpu_jpi_std]) )
                sca_values = sca_values.append( pd.Series([0.0]) )

            h = ax.bar( funcs_ps, mc_sys, width=w, label=spit_label('GT: Shared'), color=color_map[7] )
            set_barlabel( h )
            handles['gt_shared'] =  h 
            h = ax.bar( funcs_ps, mc_cpu , width=w, label=spit_label('GT: CPU'), color=color_map[6] )
            handles['gt_cpu'] =  h 


            if i == 2:
                xticks = ax.get_xticklabels()
                # xticks[3].set_text( '   ml_train' )
                # ax.set_xticklabels( xticks )

            x = ax.get_xticks()
            x = np.array(x, dtype=np.float64)

            x += offset 
            h = ax.bar( x, f23mean, width=w, label=spit_label('CD: shared'), color=color_map[15] )
            set_barlabel( h )
            handles['cd_shared'] =  h 
            h = ax.bar( x, mean_cpu, width=w, label=spit_label('CD: CPU'), color=color_map[14] )
            handles['cd_cpu'] =  h 
            #ax.errorbar( x, f23mean, f23std, fmt='o', color='Black', elinewidth=1,capthick=3,errorevery=1, alpha=0.5, ms=4, capsize=1 )
            count += 1

            x += offset
            h = ax.bar( x, fullmean, width=w, label=spit_label('Full'), color='black' )
            set_barlabel( h )
            handles['cd_full'] =  h 
            #ax.errorbar( x, fullmean, fullstd, fmt='o', color='Black', elinewidth=1,capthick=3,errorevery=1, alpha=0.5, ms=4, capsize=1 )
            count += 1

            x += offset
            h = ax.bar( x, sca_values, width=w, label=spit_label('Scaphandre'), color=color_map[0]  )
            set_barlabel( h )
            handles['sca_bars'] =  h 
            count += 1

            if i == 0:
                ax.set_ylabel( ylabel )
                saved_handles = handles

            ax.set_title( subt, fontdict={'fontsize': 9} )
            
            def get_clipped_max( x ):
                maxy = max(x)
                maxy = maxy // 10 * 10 + 10
                return maxy

            if i == 0:
                maxy = get_clipped_max(mc_sys)
                ax.yaxis.set_minor_locator(MultipleLocator(5))
                ax.set_yticks(list(np.arange(0,maxy+1,10)))
            elif i == 1:
                maxy = get_clipped_max(f23mean)
                ax.yaxis.set_minor_locator(MultipleLocator(5))
                ax.set_yticks(list(np.arange(0,maxy+1,10)))
            else:
                maxy = get_clipped_max(fullmean)
                ax.set_yticks(list(np.arange(0,maxy+1,10)))

            ax.set_ylim(0)

        txt = [ 
                'Ground Truth', 
                'Shared', 
                'CPU', 
                'Combined Disaggregation', 
                'Shared', 
                'CPU', 
                '', 
                'Pure Disaggregation', 
                'Scaphandre',
        ]

        p5 = plt.bar([0],[0], label='dummy-tophead', color='white')
        p6 = plt.bar([0],[0], label='dummy-tophead', color='white')
        p7 = plt.bar([0],[0], label='dummy-tophead', color='white')
        
        hdls = saved_handles
        self.fig.legend(
                [p5] + [hdls['gt_shared'], hdls['gt_cpu']] + 
                [p6] + [hdls['cd_shared'], hdls['cd_cpu']] + 
                [p7] + [hdls['cd_full']] + 
                [hdls['sca_bars']], 
                txt, 
                bbox_to_anchor=(0.82, 1.0),  
                ncol=3
        )
        # self.fig.legend(bbox_to_anchor=(0.8, 1.0), ncol=4)
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
    def check_if_gpu_in_dir( d ):
        dw = d.split('/')
        for w in dw:
            if is_gpu( w ):
                return True 
        return False
    dgpu = [ d for d in mc_a_dirs if check_if_gpu_in_dir(d)]
    dothers = [ d for d in mc_a_dirs if not check_if_gpu_in_dir(d)]
    mc_a_dirs = dothers

    def extract_rapl_limit( d ):
        d = d.split('/')[0]
        return d
    rapl_limits = [ extract_rapl_limit( d ) for d in mc_a_dirs ]
    
    def load_df_without_func_set( d, name ): 
        print(d+name)
        mc_sys = d+name+p_ext
        df_mc_sys = load_pickle( mc_sys )
        name = name.split('/')
        name = name[-1].split('_')[0]
        return df_mc_sys

    def load_df( d, name ): 
        df_mc_sys = load_df_without_func_set( d, name )
        df_mc_sys = df_mc_sys[ function_set ]
        return df_mc_sys

    mc_sys = []
    mc_cpu = []
    mc_shared = []
    j_per_invk_cpu = []
    j_per_invk_shared = []
    j_per_invk_full = []
    scaphandre_e_per_invk = []

    for i,d in enumerate(mc_a_dirs):
        mc_sys.append(  load_df( d, '/dfs/analysis/mc_sys' ) )
        j_per_invk_full.append(  load_df( d, '/dfs/analysis/j_per_invk_full' ) )
        
        # zero matrix to fill in for missing dataframes 
        df = mc_sys[-1].copy()
        dfz = pd.DataFrame( np.zeros( df.shape ) )
        dfz.columns = df.columns
        dfz.index = df.index
        
        def try_fill( ls, name ):
            try: 
                ls.append( load_df(d, '/dfs/analysis/'+name) )
            except FileNotFoundError:
                ls.append( dfz )

        try_fill( scaphandre_e_per_invk, 'scaphandre_e_per_invk' )
        try_fill( mc_cpu, 'mc_cpu' )
        try_fill( mc_shared, 'mc_shared' )
        try_fill( j_per_invk_cpu, 'j_per_invk_cpu' )
        try_fill( j_per_invk_shared, 'j_per_invk_shared' )
    
    if len(dgpu) != 0:
        cnn_gpu_mc = load_df_without_func_set( dgpu[0], '/dfs/analysis/mc_sys' ) 
        cnn_gpu_jpi = load_df_without_func_set( dgpu[0], '/dfs/analysis/j_per_invk_full' ) 
        cgpu = 'cnn_image_classification_gpu-0-0.0.1'
        cgpu_paper = '      ' + function_name_to_paper( cgpu ) 
        jcgpu = (cgpu_paper, cnn_gpu_mc[cgpu], cnn_gpu_jpi[cgpu].mean(), cnn_gpu_jpi[cgpu].std())

    base_dir = mc_a_dirs[0]

    plot = MCBarPlot( 1, 3, 10, 2.5, base_dir + '/dfs/plots/standalone/fig_9' )
    plot.plot_init()
    plot.plot_draw( 
                    scaphandre_e_per_invk,
                    mc_sys, 
                    mc_cpu, 
                    mc_shared, 
                    j_per_invk_cpu, 
                    j_per_invk_shared, 
                    j_per_invk_full, 
                    # ['Desktop', 'Server'], 
                    ['Desktop', 'Server', 'Jetson'], 
                    'Energy per Invoc. (Joules)', 
                    jcgpu
                  )
    plot.plot_close()

