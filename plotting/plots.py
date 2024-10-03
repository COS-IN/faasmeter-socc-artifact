
from config import *
from tags import *
from postprocessing.multiple_exps import PostProcessMultiExp
from plotting.stacked import StackPlot
from plotting.mc_lines import MCLinePlot
from plotting.mc_bars import MCBarPlot
from plotting.error_boxplot import ErrorBoxPlot 
from plotting.error_plot import ErrorPlot 

class Plots:

    def __init__(self, dirs, specific_dfs ):
        if type(dirs) != list:
            raise TypeError('Argument to Plots is not a list')
        ppmulti = PostProcessMultiExp( dirs, N_init, N, delta, specific_dfs )

        self.dfs = ppmulti.dfs
        self.base_mca = ppmulti.base_mca
        self.bases_other = ppmulti.bases_other
        self.dirs = [self.base_mca] + self.bases_other 

    def stack_plots(self):

        def _stack_plots(d):
            analysis = self.dfs[d]['analysis']
            dfs = self.dfs[d]

            if dfs[tag_dis_cpu] is not None:
                stackplot = StackPlot( 1, 1, 5, 3, d + dir_plots_stacked + '/linear_cpu' )
                stackplot.plot_init()
                df_linear_cpu = dfs[tag_dis_cpu][tags_pmins[tag_pcpu]]
                stackplot.plot_draw( analysis['all_power'][tag_pcpu], df_linear_cpu, True )
                stackplot.plot_close()

                stackplot = StackPlot( 1, 1, 5, 3, d + dir_plots_stacked + '/kf_cpu' )
                stackplot.plot_init()
                tag = 'stacked_' + tags_pmins[tag_pcpu]
                stackplot.plot_draw( analysis['all_power'][tag_pcpu], dfs[tag_dis_combined][tag], True )
                stackplot.plot_close()

                stackplot = StackPlot( 1, 1, 5, 3, d + dir_plots_stacked + '/kf_shared' )
                stackplot.plot_init()
                tag = 'stacked_' + tags_pmins[tag_pshared]
                df_kf_shared = dfs[tag_dis_combined][tag]  
                stackplot.plot_draw( analysis['all_power'][tag_pshared], df_kf_shared, True )
                stackplot.plot_close()

                stackplot = StackPlot( 1, 1, 5, 3, d + dir_plots_stacked + '/kf_f23' )
                stackplot.plot_init()
                df_kf_f23 = df_linear_cpu + df_kf_shared
                df_kf_f23 = df_kf_f23.fillna(0)
                stackplot.plot_draw( analysis['all_power'][tag_psys[0]], df_kf_f23, True )
                stackplot.plot_close()

            stackplot = StackPlot( 1, 1, 5, 3, d + dir_plots_stacked + '/kf_full' )
            stackplot.plot_init()
            tag = 'stacked_' + tags_pmins[tag_psys[0]]
            stackplot.plot_draw( analysis['all_power'][tag_psys[0]], dfs[tag_dis_combined][tag], True )
            stackplot.plot_close()

   
        for d in self.dirs:
            _stack_plots( d )
    
    def mc_line_plots(self):
        d = self.base_mca
        analysis = self.dfs[d]['analysis']
        dfs = self.dfs[d]
        
        def _mc_line_plots(tag_mc, j_per_invk, title, name):
            mc = None
            if tag_mc in dfs[tag_analysis]:
                mc = dfs[tag_analysis][tag_mc]
            funcs = j_per_invk.columns
            nplots = len(funcs)
            if 'cp' in funcs:
                nplots -= 1
            mclineplot = MCLinePlot( nplots, 1, 4, 5, d + dir_plots_mc_lines + '/' + name )
            mclineplot.plot_init()
            mclineplot.plot_draw( j_per_invk, mc, title )
            mclineplot.plot_close()
            
        if dfs[tag_dis_cpu] is not None:
            j_per_invk = dfs[tag_analysis]['j_per_invk_cpu']
            _mc_line_plots( 'mc_cpu', j_per_invk, 'CPU', 'cpu' )
            
            j_per_invk = dfs[tag_analysis]['j_per_invk_shared']
            _mc_line_plots( 'mc_shared', j_per_invk, 'Shared', 'shared' )
            
            # F23
            j_per_invk = dfs[tag_analysis]['j_per_invk_cpu'] + dfs[tag_analysis]['j_per_invk_shared']
            _mc_line_plots( 'mc_sys', j_per_invk, 'F23', 'f23' )
        
         # F22
        j_per_invk = dfs[tag_analysis]['j_per_invk_full']
        _mc_line_plots( 'mc_sys', j_per_invk, 'F22', 'f22' )

    def mc_bar_plots(self):
        d = self.base_mca
        analysis = self.dfs[d]['analysis']
        dfs = self.dfs[d]

        def _mc_bar_plots(tag_mc, j_per_invk, title, name, df_sca_pinvk=None ):
            mc = dfs[tag_analysis][tag_mc]

            funcs = mc.columns
            mcbarplot = MCBarPlot( 1, 1, 5, 3, d + dir_plots_mc_bars + '/' + name )
            mcbarplot.plot_init()
            mcbarplot.plot_draw( j_per_invk, mc, title, df_sca_pinvk )
            mcbarplot.plot_close()
        
        stag = 'scaphandre_e_per_invk' 

        if stag in analysis:
            df_sca_pinvk = analysis[stag]
        else:
            df_sca_pinvk = None

        if dfs[tag_dis_cpu] is not None:
            j_per_invk = dfs[tag_analysis]['j_per_invk_cpu']
            _mc_bar_plots( 'mc_cpu', j_per_invk, 'CPU', 'cpu', df_sca_pinvk )

            j_per_invk = dfs[tag_analysis]['j_per_invk_shared']
            _mc_bar_plots( 'mc_shared', j_per_invk, 'Shared', 'shared' )

            # F23
            j_per_invk = dfs[tag_analysis]['j_per_invk_cpu'] + dfs[tag_analysis]['j_per_invk_shared']
            _mc_bar_plots( 'mc_sys', j_per_invk, 'F23', 'f23', df_sca_pinvk )

        # F22
        j_per_invk = dfs[tag_analysis]['j_per_invk_full']
        _mc_bar_plots( 'mc_sys', j_per_invk, 'F22', 'f22', df_sca_pinvk )
    
    def errors_gen_plots(self, Class, prefix): 
        d = self.base_mca
        analysis = self.dfs[d]['analysis']

        def _error_plots(errors_per_func, title, name):
            errorplot = Class( 1, 1, 4, 5, d + dir_plots_mc_lines + '/' + name )
            errorplot.plot_init()
            errorplot.plot_draw( errors_per_func, title )
            errorplot.plot_close()

        if self.dfs[d][tag_dis_cpu] is not None:
            _error_plots( analysis[prefix+'cpu'], 'CPU', prefix+'cpu' )
            _error_plots( analysis[prefix+'shared'], 'Shared', prefix+'shared' )
            _error_plots( analysis[prefix+'sys_f23'], 'F23', prefix+'f23' )
        _error_plots( analysis[prefix+'sys_full'], 'F22', prefix+'f22' )

    def error_plots(self):
        self.errors_gen_plots( ErrorPlot, 'errors_per_func_' )

    def error_boxplots(self):
        self.errors_gen_plots( ErrorBoxPlot, 'errors_all_' )
        self.errors_gen_plots( ErrorBoxPlot, 'errors_mean_' )

    def plot_everything(self):
        self.stack_plots()
        self.mc_line_plots()
        self.mc_bar_plots()
        self.error_boxplots()
        self.error_plots()


