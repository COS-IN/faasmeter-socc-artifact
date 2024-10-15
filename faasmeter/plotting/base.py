import os 

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class PlotBase:

    def __init__(self, rows, cols, width, height, name):
        self.rows = rows 
        self.cols = cols
        self.name = name
        self.width = width
        self.height = height

        os.system('mkdir -p ' + os.path.dirname(name))

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
        # matplotlib.rcParams['text.usetex'] = True

    def init_genaxis(self, ):
        plt.close()
        fig, axs = plt.subplots( 
                                self.rows, 
                                self.cols, 
                                figsize=(self.width,self.height), 
                                sharex=True, 
                                sharey=False 
        )
        
        self.fig = fig
        self.axs = axs

        w = 0.2
        self.bar_w = w
        self.bar_offset = w + 0.03
        self.count = 0

    def plot_init(self):
        self.init_setrcParams()
        self.init_genaxis()

    def plot_draw(self):
        pass

    def plot_close(self):
        plt.show()
        os.system('rm ' + self.name + '.* > /dev/null 2>&1')
        self.fig.savefig( self.name + '.png', bbox_inches='tight') 
        self.fig.savefig( self.name + '.pdf', bbox_inches='tight' ) 
        print('Saved: {}'.format(self.name))
        plt.close()

    def get_color_scheme(self, colors):
        if colors == "brc4":
            brc=['#edf8b1', '#7fcdbb', '#2c7fb8', 'black']
        elif colors == "brc":
            brc=['#8dd3c7', '#ffffb3', '#bebada', 'black']
        elif colors == "brc2":
            #Teal yellow purple. First and third very similar http://colorbrewer2.org/?type=qualitative&scheme=Set3&n=3
            brc=['#fc8d59','#ffffbf', '#99d594','black']
        elif colors == "brc4":
            #Orange Yellow Green. BEST. http://colorbrewer2.org/?type=diverging&scheme=Spectral&n=3
            brc=['#f1a340','#fc8d59' , '#998ec3', '0.6', 'black']
        elif colors == "pulse":
            #Orange Yellow Green. BEST. http://colorbrewer2.org/?type=diverging&scheme=Spectral&n=3
            brc=['orange', 'tab:cyan', 'tab:green', 'tab:gray', 'tab:purple', 'tab:brown', 'tab:olive', 'tab:blue', 'pink']
        elif colors == "additive":
            #Orange Yellow Green. BEST. http://colorbrewer2.org/?type=diverging&scheme=Spectral&n=3
            brc=['orange', 'pink', 'tab:green', 'tab:gray', 'tab:purple', 'tab:brown', 'tab:olive', 'tab:blue', 'tab:cyan', 'gold', 'maroon', 'steelblue']
        else:
            raise Exception(f"unknown color scheme '{colors}'")
        return brc

