import numpy as np
import pandas as pd

from plotting.base import PlotBase
from helper_funcs import *

class ErrorPlot(PlotBase):

    def plot_draw( self, errors_per_func, title ):
        funcs = errors_per_func.columns
        funcs_p = [ function_name_to_paper(f) for f in funcs ]

        ax = self.axs
        ax.boxplot( errors_per_func, labels=funcs_p, showfliers=False )
        ax.set_ylabel( 'Error %' )
        self.fig.suptitle( title )
