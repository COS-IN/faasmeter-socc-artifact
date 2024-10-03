from plotting.base import PlotBase
from helper_funcs import *

class StackPlot(PlotBase):

    def plot_draw( self, agg, subs, start_zero ):
        agg = agg.copy()
        subs = subs.copy()
        ax = self.axs
        if start_zero:
            start = agg.index[0]
            agg.index -= start
            subs.index -= start
        if 'cp' in subs.columns:
            cols = [ c for c in subs.columns if c != 'cp' ]
            subs = subs[cols]
        labels = [ function_name_to_paper(c) for c in subs.columns ]
        ys = []
        for c in subs.columns:
            ys.append( subs[c] )
        ax.stackplot(
                    subs.index,
                    *ys,
                    labels=labels,
                    baseline='zero',
                    colors=self.get_color_scheme('pulse')
        )
        ax.plot( agg.index, agg, label=agg.name, linestyle="--", color='red' )
        ax.set_xlabel('Time (mins)')
        ax.set_ylabel('Power (Watts)')
        self.fig.legend(ncol=len(subs.columns)+1, bbox_to_anchor=(1.0, 1.0))
