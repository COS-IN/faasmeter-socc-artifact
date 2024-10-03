import numpy as np
import pandas as pd

from plotting.base import PlotBase
from helper_funcs import *

class ErrorBoxPlot(PlotBase):

    def plot_draw( self, errors_all, title ):
        ax = self.axs
        ax.boxplot( errors_all, showfliers=False )
        ax.set_ylabel( 'Error %' )
        self.fig.suptitle( title )
