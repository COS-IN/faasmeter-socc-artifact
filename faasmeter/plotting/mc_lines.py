from plotting.base import PlotBase
from helper_funcs import *

class MCLinePlot(PlotBase):

    def plot_draw( self, j_per_invk, mc_truth, title ):
        funcs = [ f for f in j_per_invk.columns if f != 'cp' ]
        funcs_p = [ function_name_to_paper(f) for f in funcs if f != 'cp' ]

        axs = self.axs
        for i,f in enumerate(funcs):
            ax = axs[i]
            x = j_per_invk.index.copy()
            x -= x[0]
            if f in j_per_invk.columns:
                y = j_per_invk[f]
                ax.plot( x, y )
            if mc_truth is not None:
                yt = [ mc_truth[f][0] for i in x ]
                ax.plot( x, yt, linestyle='--' )
            ax.set_ylabel( funcs_p[i] )
            ax.set_ylim(0)
        ax.set_xlabel('Time (mins)')
        self.fig.suptitle( title )
