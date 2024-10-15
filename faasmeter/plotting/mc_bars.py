import numpy as np # for np.array only! 

from plotting.base import PlotBase
from helper_funcs import *

class MCBarPlot(PlotBase):

    def plot_draw( self, j_per_invk, mc_truth, title, df_sca_pinvk=None ):
        funcs = mc_truth.columns
        funcs = j_per_invk.columns.intersection(funcs)
        funcs_p = [ function_name_to_paper(f) for f in funcs ]

        j_per_invk_mean = j_per_invk[funcs].mean()
        j_per_invk_std = j_per_invk[funcs].std()

        w = 0.2
        offset = w + 0.02
        count = 1 
        ax = self.axs
        ax.bar( funcs_p, j_per_invk_mean, width=w, label='Measurement' )
        try:
            ax.errorbar( funcs_p, j_per_invk_mean, j_per_invk_std, fmt='.', color='Black', elinewidth=2,capthick=10,errorevery=1, alpha=0.5, ms=4, capsize = 2 )
        except StopIteration:
            print("Error: drawing errorbars in MC bar plot")
            print("       j_per_invk_mean: {}".format( j_per_invk_mean ))
            print("       j_per_invk_std: {}".format( j_per_invk_std ))
            exit(-1)

        x = ax.get_xticks()
        x = np.array(x, dtype=np.float64)
        
        if df_sca_pinvk is not None:
            x += offset 
            ax.bar( x, df_sca_pinvk[funcs].loc[0], width=w, label='Scaphandre' )
     
        x += offset 
        ax.bar( x, mc_truth[funcs].loc[0], width=w, label='Ground Truth' )
        
        ax.legend(loc='upper left')
        self.fig.suptitle( title )

