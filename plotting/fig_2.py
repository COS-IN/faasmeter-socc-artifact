import pickle
from copy import deepcopy
from pprint import pprint
import re 
import sys 
import matplotlib.patches as mpatches
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy import integrate
import matplotlib.pyplot as plt
from faasmeter.faasmeter.parsing.Logs_to_df import *

############################################################################
# Saving and Restoring Variables using pickle for faster rexecutions of the
# script
############################################################################

def save_as_pkl(var):
    """
    var is the variable to be saved as pickle 
    """
    gvars = globals()

    var_name = [k for k, v in gvars.items() if id(v) == id(var)]
    if len(var_name) == 0:
        lvars = locals()
        var_name = [k for k, v in lvars.items() if id(v) == id(var)][0]
    else:
        var_name = var_name[0]

    with open(var_name+".pkl", 'wb') as f:
        pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)

def restore_pkl(d):
    """
    d is the variable name that needs to be restored from a pickle file with same name 
    """
    try:
        with open(d+".pkl", 'rb') as f:
            b = pickle.load(f)
            gvars = globals()
            gvars[d] = b

            return True
    except FileNotFoundError:
        return False

    return False

def load_pkl_file( fname ):
    try:
        with open(fname, 'rb') as f:
            b = pickle.load(f)
        
            return b

    except FileNotFoundError:
        return None

    return None




#################################################################
# Fetch Data
##################################################################

power_source = ['igpm', 'ipmi', 'bat' ]
#if True: # Analysis on desktop data 
if False: # Analysis on desktop data 
    power_source = power_source[0]
    mc_type = 'desktop'
    
    minsights = "../marginal_cost/desktop/insights.pkl"
    minsights = load_pkl_file( minsights )

    base_dir = "../../experiments/desktop/singelton-benchmark"
    
    funcs = [
        "chameleon",
        "float_operation",
        "hello",
        "lin_pack",
    ]
    funcs = [
        "dd",
        "pyaes",
        "gzip_compression",
        "json_dumps_loads",
        "image_processing",
        "model_training",
        "video_processing",
        "cnn_image_classification",
    ]

    thread_count = [1,4,8]
    thread_count = [str(t) for t in thread_count ]
    
    exceptions = [
        "chameleon-4",
        "chameleon-8",
        "float_operation-4",
        "float_operation-8",
        "hello-4",
        "hello-8",
        "lin_pack-4",
        "lin_pack-8",
    ]

else: # Analysis on victor data 
    power_source = power_source[1]
    mc_type = 'server'

    minsights = "../../results/marginal_cost/victor/insights.pkl"
    minsights = load_pkl_file( minsights )

    base_dir = "../../results/singelton-benchmark"
    funcs = [
        "chameleon",
        "float_operation",
        "hello",
        "lin_pack",
    ]
    funcs = [
        "dd",
        "pyaes",
        "gzip_compression",
        "json_dumps_loads",
        "image_processing",
        "model_training",
        "video_processing",
        "cnn_image_classification",
    ]

    thread_count = [1,4,8,16]
    thread_count = [1,4,8]
    thread_count = [str(t) for t in thread_count ]
    
    exceptions = [
        "chameleon-4",
        "chameleon-8",
        "chameleon-16",
        "float_operation-4",
        "float_operation-8",
        "float_operation-16",
        "hello-4",
        "hello-8",
        "hello-16",
        "lin_pack-4",
        "lin_pack-8",
        "lin_pack-16",
    ]

func_dirs = []
for f in funcs:
    for t in thread_count:
        tf =  f + "-" + t
        if tf in exceptions:
            continue

        d = base_dir + "/" + tf + "/" 
        func_dirs.append( d )

if not (restore_pkl("ologs")):
 
    def rename_power_source( lg ):
        #lg.power_df = lg.power_df.rename( columns={'perf_rapl':'rapl'} )
        #lg.power_df = lg.power_df.rename( columns={power_source:'power'} )
        pass
   
    def print_data( lg ):
        pass
        # print(f"Columns: {lg.worker_df.columns}")
        # print(lg.worker_df)
        # print(f"Columns: {lg.power_df.columns}")
        # print(lg.power_df)

    ologs = [ Logs_to_df( d, mc_type ) for d in func_dirs ]
    
    print("###########################")
    print("Processing")
    pprint( func_dirs ) 
    print("############")
    for l in ologs:
        l.process_all_logs()
        rename_power_source( l )
        print_data( l )
    
    save_as_pkl(ologs)
    """
    Worker DF: Columns: Index(['fn_start', 'fn_end', 'tid', 'fqdn'], dtype='object')
    Power DF: Columns: Index(['cpu_pct_process', 'cpu_time', 'load_avg_1minute', 'cpu_pct',
       'hw_cpu_hz_mean', 'hw_cpu_hz_max', 'hw_cpu_hz_min', 'hw_cpu_hz_std',
       'hw_cpu0_hz', 'hw_cpu1_hz', 'hw_cpu2_hz', 'hw_cpu3_hz', 'hw_cpu4_hz',
       'hw_cpu5_hz', 'hw_cpu6_hz', 'hw_cpu7_hz', 'hw_cpu8_hz', 'hw_cpu9_hz',
       'hw_cpu10_hz', 'hw_cpu11_hz', 'kern_cpu_hz_mean', 'kern_cpu_hz_max',
       'kern_cpu_hz_min', 'kern_cpu_hz_std', 'kern_cpu0_hz', 'kern_cpu1_hz',
       'kern_cpu2_hz', 'kern_cpu3_hz', 'kern_cpu4_hz', 'kern_cpu5_hz',
       'kern_cpu6_hz', 'kern_cpu7_hz', 'kern_cpu8_hz', 'kern_cpu9_hz',
       'kern_cpu10_hz', 'kern_cpu11_hz', 'rapl', 'energy_ram',
       'retired_instructions', 'power', 'voltage', 'current', 'power_factor',
       'discard'],
      dtype='object')
    """


#################################################################
# Global Data 
##################################################################
insights = {
   'func': {
        'thread_count' : {
            'start': 0,
            'end': 0,
            'total_invocations': 0,
            'kf_rdict': None,
            'kf_rdict_full': None,
            'avg':{
                'src':{
                    'eng': 0,
                    'power': 0,
                },
                'rapl':{
                    'eng': 0,
                    'power': 0,
                },
            },
            'per_inv':{
                'exec_time_avg': 0,
                'exec_time_var': 0,
                'samples_count_avg': 0,
                'samples_count_var': 0,
                'src':{
                    'eng':{
                        'mean': 0,
                        'var': 0,
                    },
                    'power':{
                        'mean': 0,
                        'var': 0,
                    }
                },
                'rapl':{
                    'eng':{
                        'mean': 0,
                        'var': 0,
                    },
                    'power':{
                        'mean': 0,
                        'var': 0,
                    }
                },
            },
        },
    },
}

for f in funcs:
    
    insights[f] = deepcopy( insights['func'] )
    insights[f].pop( 'thread_count' )

    for t in thread_count:
        tf =  f + "-" + t
        if tf in exceptions:
            continue

        insights[f][t] = deepcopy( insights['func']['thread_count'] )

#################################################################
# Global Functions
##################################################################
def select_func( wdf, func ):
    return wdf[wdf.fqdn.str.match(func+'.*')]

def total_invocations( wdf, func ):
    return len( select_func( wdf, func ) )

def integrate_col(df, col_name):
    df = pd.DataFrame( df[col_name] )
    df = df.rename( columns={0:col_name} )
    df['timestamp'] = df.index.to_series() 
    df['time_delta'] = df['timestamp'].diff().dt.total_seconds()
    df['time_intg'] = df['time_delta'].cumsum()
    df = df.dropna()
    
    energy = integrate.trapz(y=df[col_name], x=df['time_intg'])

    df = df.drop( columns=['timestamp','time_delta','time_intg'] )

    return energy

def construct_exec_time_col( wdf ):
    def exec_time( row ):
        return (row['fn_end'] - row['fn_start']).total_seconds()

    wdf['exec_time'] = wdf.apply( exec_time, axis=1 )

        
#################################################################
# Analyze Function and Thread data - for invocation matrice  
##################################################################

if not (restore_pkl("insights")):
    i = -1
    for f in funcs:
        for t in thread_count: 
            tf =  f + "-" + t
            if tf in exceptions:
                continue
            i += 1
 
            #################################################################
            # Generating rdict for each function 
            ##################################################################
  
            edata = ologs[i]

            edata.output_type = "indiv"
            edata.collapse_fqdns = True
            edata.should_collapse_similar_funcs = False
            edata.populate_system_settings()
            edata.process_all_logs()

            x = edata.init_estimates()
            rdict = edata.pow_full_breakdown( x )
            insights[f][t]['kf_rdict'] = rdict

            edata.output_type = "full"
            edata.collapse_fqdns = True
            edata.populate_system_settings()
            edata.process_all_logs()

            x = edata.init_estimates()
            rdict = edata.pow_full_breakdown( x )
            insights[f][t]['kf_rdict_full'] = rdict

            pdf = ologs[i].power_df 
            wdf = ologs[i].worker_df
            power_means = ologs[i].get_realpower_means()

            # for indexing ologs since they were loaded in exact same order 

            insights[f][t]['start'] = wdf.iloc[0]['fn_start']
            insights[f][t]['end']   = wdf.iloc[-1]['fn_end']
            insights[f][t]['total_invocations'] = total_invocations( wdf, f )
            
            construct_exec_time_col( wdf )
            
            insights[f][t]['per_inv']['exec_time_avg'] = wdf['exec_time'].mean()
            insights[f][t]['per_inv']['exec_time_var'] = wdf['exec_time'].var()
            
            spdf = pdf[ insights[f][t]['start'] : insights[f][t]['end'] ]
            
            insights[f][t]['avg']['src']['eng'] = integrate_col( spdf, power_means ) / insights[f][t]['total_invocations'] 
            insights[f][t]['avg']['src']['power'] = insights[f][t]['avg']['src']['eng'] / insights[f][t]['per_inv']['exec_time_avg'] 
            
            insights[f][t]['avg']['rapl']['eng'] = integrate_col( spdf, 'perf_rapl' ) / insights[f][t]['total_invocations'] 
            insights[f][t]['avg']['rapl']['power'] = insights[f][t]['avg']['rapl']['eng'] / insights[f][t]['per_inv']['exec_time_avg'] 

            if insights[f][t]['per_inv']['exec_time_avg'] < 0.250:
                print(f"Warning we cannot perform singlenton analysis on {f}-{t} because it's execution time is too small {insights[f][t]['per_inv']['exec_time_avg']}")
                continue

            # Per invocation analysis 
     
            inv_s_j = []
            inv_r_j = []
            inv_s_p = []
            inv_r_p = []
            sample_count = []
            for inv in range(len(wdf)):
                r = wdf.iloc[inv]
                ts = r['fn_start']
                te = r['fn_end']

                sdf = pdf[  ts : te ] # it can contain overlapping invocations 
                
                sample_count.append( len(sdf) )
                inv_s_j.append( integrate_col( sdf, power_means ) )
                inv_r_j.append( integrate_col( sdf, 'perf_rapl' ) )

                # compensating overlapping invocations 
                rfns = wdf[~(wdf['fn_end'] < ts) & ~(wdf['fn_start'] > te)]
                inv_s_j[-1] = inv_s_j[-1] / len(rfns)
                inv_r_j[-1] = inv_r_j[-1] / len(rfns)

                # calculating power
                sj = inv_s_j[-1]
                rj = inv_r_j[-1]
                exec_t = (te-ts).total_seconds()
                
                inv_s_p.append( sj / exec_t )
                inv_r_p.append( rj / exec_t )

            
            def save_mean_var( lhs, ls, lsk ):
                invdf = pd.DataFrame( ls )
                lhs[lsk[0]] = invdf[0].mean()
                lhs[lsk[1]] = invdf[0].var()
                
            save_mean_var( insights[f][t]['per_inv']                  , sample_count, ['samples_count_avg', 'samples_count_var'] )
            save_mean_var( insights[f][t]['per_inv']['src']['eng']    , inv_s_j     , ['mean', 'var'] )
            save_mean_var( insights[f][t]['per_inv']['rapl']['eng']   , inv_r_j     , ['mean', 'var'] )
            save_mean_var( insights[f][t]['per_inv']['src']['power']  , inv_s_p     , ['mean', 'var'] )
            save_mean_var( insights[f][t]['per_inv']['rapl']['power'] , inv_r_p     , ['mean', 'var'] )
    save_as_pkl(insights)

###################################################################################
# Plotting insights 
###################################################################################

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams['axes.linewidth'] = 0.5
matplotlib.rcParams['axes.labelpad'] = 2.0
matplotlib.rcParams['axes.titlepad'] = 2.0
matplotlib.rcParams['figure.dpi'] = 200.0
matplotlib.rcParams['figure.subplot.left'] = 0.2

class BarPlot():
    width = 0.5  # bar width
    gap = 0.5    # gap between groups
    gap_bar = 0.1 # gap between bars within a group 
    _xlocs = 0.5 # start location of first bar

    def __init__(self, w, h):
        
        fig, ax = plt.subplots()
        fig.set_size_inches( w, h )

        self.fig = fig 
        self.ax = ax 

    def __populate_bar_given_col( self, data: list, bar_count: int ):
        """
        Assumptions
            number of groups will not change - len(data)
        Input
            data -> list of elements, each element corresponds to 
                    height bar_count should have in each 
                    group of bars 
            bar_count -> bar number for these entries 
        Output
            (x,y) tuple of list of x,y coordinates 
        """
        y = []
        x = []
        # for each start and end in the targets_dic 
        for i,d in enumerate(data):
            y.append( d )
            xcord =  self._xlocs + i * self.items * (self.width + self.gap_bar) + i * self.gap + bar_count * (self.width + self.gap_bar)
            x.append( xcord )
        
        return (x,y)


    def write_bar(self, groups: list, ylabel: str, dic: dict ) -> None:
        """
        Assumptions
            
        Input
            groups -> list of group names  
            ylabel -> description of y axis 
            dic -> dictionary of bar name against list of heights 
                   for each group 
        Output
            state change of plt - to have drawn bars on it 
        """

        #map of color for the bars 
        #cmap(i) would generate the color 
        cmap = plt.cm.get_cmap('hsv', len(dic) + 1)
        
        self.items = len(dic)
        i = 0
        for k,v in dic.items():
            xys = self.__populate_bar_given_col( dic[k], i )
            self.ax.bar( xys[0], xys[1], label=k, color=cmap(i), width=self.width )

            i += 1

        labels_xloc = [ x - (self.items*self.width)/2 + self.width/2 for x in xys[0] ]

        plt.xticks( labels_xloc, groups, rotation='vertical' )
        plt.ylabel(ylabel)

    def write_bar_with_bottoms(self, groups: list, ylabel: str, dics: list, styles: list = None ) -> None:
        """
        Assumptions
            
        Input
            groups -> list of group names  
            ylabel -> description of y axis 
            dics -> list of dictionaries to be stacked against list of heights 
                   for each group 
        Output
            state change of plt - to have drawn bars on it 
        """

        tbars = len(dics[0])
        tstacks = len(dics)

        #map of color for the bars 
        #cmap(i) would generate the color 
        cmap = plt.cm.get_cmap('hsv', tstacks*tbars + 1)
        
        def get_color_index( bar_id, stack_id ):
            return bar_id + stack_id 

        # generating x coordinates
        dic = dics[0]
        self.items = len(dic)
        i = 0
        for k,v in dic.items():
            xys = self.__populate_bar_given_col( dic[k], i )

            #self.ax.bar( xys[0], xys[1], label=k, color=cmap(i), width=self.width )
            #self.ax.bar( xys[0], dics[1][k], label=k, color=cmap(i+1), width=self.width, bottom = xys[1] )
            #self.ax.bar( xys[0], dics[2][k], label=k, color=cmap(i+2), width=self.width, bottom = np.array(xys[1]) + np.array(dics[1][k])  )
            
            def select_if_available( k, d ):
                return d[k] if k in d else ''
            
            h = ['|||','///','---'] 
            hs = [ [h] * len(v) for h in h ] 

            old_h = None
            j = 0
            for d in dics:
                if old_h is None:
                    #self.ax.bar( xys[0], d[k], width=self.width, hatch=select_if_available( 'hatch', styles[j] ), color=styles[j]['color'], edgecolor='black' )
                    #self.ax.bar( xys[0], d[k], width=self.width, hatch=select_if_available( 'hatch', styles[j] ), color=styles[j]['color'] )
                    self.ax.bar( xys[0], d[k], width=self.width, hatch=hs[i], color=styles[j]['color'], alpha=0.99 )
                    old_h = d[k]
                else:
                    #self.ax.bar( xys[0], d[k], width=self.width, bottom = old_h, hatch=select_if_available( 'hatch', styles[j] ) , color=styles[j]['color'], edgecolor='black' )
                    #self.ax.bar( xys[0], d[k], width=self.width, bottom = old_h, hatch=select_if_available( 'hatch', styles[j] ) , color=styles[j]['color'] )
                    self.ax.bar( xys[0], d[k], width=self.width, bottom = old_h, hatch=hs[i] , color=styles[j]['color'], alpha=0.99 )
                    old_h = np.array( old_h ) + np.array( d[k] )

                j += 1

            i += 1

        labels_xloc = [ x - (self.items*self.width)/2 + self.width/2 for x in xys[0] ]
        
        pprint(labels_xloc)
        pprint(groups)
        #plt.xticks( labels_xloc, groups, rotation='vertical' )
        plt.xticks( labels_xloc, groups, rotation='horizontal' )
        plt.ylabel(ylabel)


    def save_fig(self, fig_name: str ):
        #self.ax.legend()
        print("SAVING BarPlot", fig_name)
        plt.savefig( fig_name )

class LinePlot():
    linewidth = 2 

    def __init__(self, w, h):
        
        fig, ax = plt.subplots()
        fig.set_size_inches( w, h )

        self.fig = fig 
        self.ax = ax 

    def __populate_bar_given_col( self, data: list, bar_count: int ):
        """
        Assumptions
            number of groups will not change - len(data)
        Input
            data -> list of elements, each element corresponds to 
                    height bar_count should have in each 
                    group of bars 
            bar_count -> bar number for these entries 
        Output
            (x,y) tuple of list of x,y coordinates 
        """
        y = []
        x = []
        # for each start and end in the targets_dic 
        for i,d in enumerate(data):
            y.append( d )
            xcord =  self._xlocs + i * self.items * self.width + i * self.gap + bar_count * self.width 
            x.append( xcord )
        
        return (x,y)
    
    def __sanitize_data( self, x, y ):
        """
        If y has lower dimension append zeros until the dimensions match.
        """

        while len(y) < len(x):
            y.append(0)

    def write_line(self, x: list, dic_y: dict, x_axis_label: str, y_axis_label: str ) -> None:
        """
        Assumptions
            
        Input
            x -> coordinates for x axis 
            dic_y -> dictionary of y coordinates - keys are to be used as labels 
            x_axis_label -> label to be used for x axis 
            y_axis_label -> label to be used for y axis 
        Output
            state change of plt - to have drawn lines on it  
        """

        #map of color for the bars 
        #cmap(i) would generate the color 
        cmap = plt.cm.get_cmap('hsv', len(dic_y) + 1)
        
        self.items = len(dic_y)
        i = 0
        for k,v in dic_y.items():
            self.__sanitize_data( x, v )
            self.ax.plot( x, v, label=k, color=cmap(i), linewidth=self.linewidth ) 

            i += 1

        plt.xlabel( x_axis_label )
        plt.ylabel( y_axis_label )

    def save_fig(self, fig_name: str ):
        self.ax.legend()
        print("SAVING LinePlot", fig_name)
        plt.savefig( fig_name )

# width / height
ratio = 2
h = 14

groups = []
i = -1
for f in funcs:
    for t in thread_count: 
        tf =  f + "-" + t
        if tf in exceptions:
            continue
        i += 1
 
        groups.append(tf)

###
# Plotting energy per invocation - from rdict - individual  

plt.close()

fig, ax=plt.subplots()
fig.set_size_inches(16,8)

labels = []
Energy = []
Cplane = []
Idle = []

for f in funcs:
    for t in thread_count: 
        tf =  f + "-" + t
        tfs = function_name_to_paper( f ) + '-' + t 
        if tf in exceptions:
            continue
        i += 1
        
        rdict = insights[f][t]['kf_rdict']
        
        labels.append( tfs )
        Energy.append( rdict["Energy"][0] )
        Cplane.append( rdict["per_invok_cp"][0] )
        Idle.append( rdict["per_invok_idle"][0] )

ax.bar(labels, Energy, label="Individual")
ax.bar(labels, Cplane, bottom = Energy, label="Control Plane Share")
ax.bar(labels, Idle, bottom = np.array(Energy)+np.array(Cplane), label="Idle Share")

ax.set_ylabel("Energy per invocation (J)")
ax.legend()

plt.xticks( rotation='vertical' )
print("SAVING ", "fig_2_lgs_to_df_indv.jpg")
plt.savefig( 'fig_2_lgs_to_df_indv.jpg' )

###
# Plotting energy per invocation - from rdict individual using my bar method  


def plot_fig_2_smart( funcs, fig_name ):
    plt.close()

    # width / height
    w = 5
    h = 3

    bpl = BarPlot(w, h)
    bpl.width = 0.015
    bpl.gap =   0.005
    bpl.gap_bar =   0.002

    dic_energy = {
        'thread_1':  [],
        'thread_4':  [],
        'thread_8':  [],
    }
    dic_cplane = {
        'thread_1':  [],
        'thread_4':  [],
        'thread_8':  [],
    }
    dic_idle = {
        'thread_1':  [],
        'thread_4':  [],
        'thread_8':  [],
    }

    labels = []

    for f in funcs:
        for t in thread_count: 
            tf =  f + "-" + t
            tfs = function_name_to_paper( f ) + '-' + t 
            if tf in exceptions:
                continue
            
            rdict = insights[f][t]['kf_rdict']
            
            fpn = function_name_to_paper(f)
            if fpn not in labels:
                labels.append( fpn )
            dic_energy['thread_'+t].append( rdict["Energy"][0] )
            dic_cplane['thread_'+t].append( rdict["per_invok_cp"][0] )
            dic_idle['thread_'+t].append( rdict["per_invok_idle"][0] )

    pprint(labels)
    pprint(dic_energy)
    styles = [
        {
            #'hatch':'-|',
            'color':'#2c7fb8'
        },
        {
            'hatch':'|||',
            'color':'#7fcdbb'
        },
        {
            'hatch':'---',
            'color':'#edf8b1'
        },
    ]
    bpl.write_bar_with_bottoms( labels, "Energy per invocation (J)", [dic_energy, dic_cplane, dic_idle], styles )
    bpl.save_fig( fig_name )

plot_fig_2_smart( funcs[4:], 'fig_2_smart_large_funcs.jpg' )
plot_fig_2_smart( funcs[:4], 'fig_2_smart_small_funcs.jpg' )
plot_fig_2_smart( funcs, 'fig_2_smart_all_funcs.jpg' )

plot_fig_2_smart( funcs[4:], 'fig_2_smart_large_funcs.pdf' )
plot_fig_2_smart( funcs[:4], 'fig_2_smart_small_funcs.pdf' )
plot_fig_2_smart( funcs, 'fig_2_smart_all_funcs.pdf' )

###
#Drawing Legend
plt.close()

colors = ['#2c7fb8' , '#7fcdbb', '#edf8b1']
labelsc = ["Individual", "Control Plane", "Idle" ]
hatches = ['|||','///','---'] 
labelsh = ["Threads-1", "Threads-4", "Threads-8" ]

#f = lambda m,c: plt.plot([],[], marker=m, color=c, ls="none")[0]
f = lambda m,c: mpatches.Patch( alpha=0.99,color=c,ls='none')
#h = lambda m,c: plt.bar([],[], hatch=c, ls="none")
h = lambda m,c: mpatches.Patch( alpha=0.99,hatch=c, facecolor='w', edgecolor='0.0',ls='none')
handles = [f("s", colors[i]) for i in range(3)]
handlesh = [h("s", hatches[i]) for i in range(3)]

handles = handles + handlesh
labels = labelsc + labelsh  
legend = plt.legend(handles, labels, loc=3, framealpha=1,  ncol=6, bbox_to_anchor=(0.5,1.35))
# frameon=False,
def export_legend(legend, filename="legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    print("SAVING legend", filename)

    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)
export_legend(legend, 'legend.jpg')
#plt.show()

###
# Plotting energy per invocation - from rdict - full 

plt.close()

fig, ax=plt.subplots()
fig.set_size_inches(16,8)

labels = []
Energy = []
Cplane = []
Idle = []

for f in funcs:
    for t in thread_count: 
        tf =  f + "-" + t
        tfs = function_name_to_paper( f ) + '-' + t 
        if tf in exceptions:
            continue
        i += 1
        
        rdict = insights[f][t]['kf_rdict_full']
        
        labels.append( tfs )
        Energy.append( rdict["Energy"][0] )

ax.bar(labels, Energy, label="Full")

ax.set_ylabel("Energy per invocation (J)")
ax.legend()

plt.xticks( rotation='vertical' )
print("SAVING ", "fig_2_lgs_to_df_full.jpg")
plt.savefig( 'fig_2_lgs_to_df_full.jpg' )

###
#Plotting Energy Per Invocation - Average
h = 4
plt.close()
bpl = BarPlot(h*ratio,h)

dic = {
    power_source+'_avg':  [],
#    power_source+'_per_inv':  [],
#    power_source+'_var':  [],
    'rapl_avg': [],
#    'rapl_per_inv': [],
#    'rapl_var': [],
}

i = -1
for f in funcs:
    for t in thread_count: 
        tf =  f + "-" + t
        if tf in exceptions:
            continue
        i += 1
 
        dic[power_source+'_avg'].append(     insights[f][t]['avg']['src']['eng'] )
#        dic[power_source+'_per_inv'].append( insights[f][t]['per_inv']['src']['eng']['mean'] )
#        dic[power_source+'_var'].append(     insights[f][t]['per_inv']['src']['eng']['var'] )

        dic['rapl_avg'].append(     insights[f][t]['avg']['rapl']['eng'] )
#        dic['rapl_per_inv'].append( insights[f][t]['per_inv']['rapl']['eng']['mean'] )
#        dic['rapl_var'].append( insights[f][t]['per_inv']['rapl']['eng']['var'] )

bpl.write_bar( groups, 'Energy (Joules) Per Invocation', dic )
bpl.save_fig( 'figure_2_energy.jpg' )

###
#Plotting Power Per Invocation

plt.close()
bpl = BarPlot(h*ratio,h)

dic = {
    power_source+'_avg':  [],
#    power_source+'_per_inv':  [],
#    power_source+'_var':  [],
    'rapl_avg': [],
#    'rapl_per_inv': [],
#    'rapl_var': [],
}

i = -1
for f in funcs:
    for t in thread_count: 
        tf =  f + "-" + t
        if tf in exceptions:
            continue
        i += 1
 
        dic[power_source+'_avg'].append(     insights[f][t]['avg']['src']['power'] )
#        dic[power_source+'_per_inv'].append( insights[f][t]['per_inv']['src']['power']['mean'] )
#        dic[power_source+'_var'].append(     insights[f][t]['per_inv']['src']['power']['var'] )

        dic['rapl_avg'].append(     insights[f][t]['avg']['rapl']['power'] )
#        dic['rapl_per_inv'].append( insights[f][t]['per_inv']['rapl']['power']['mean'] )
#        dic['rapl_var'].append( insights[f][t]['per_inv']['rapl']['power']['var'] )

bpl.write_bar( groups, 'Power (Watts) Per Invocation', dic )
bpl.save_fig( 'figure_2_power.jpg' )

###
#Plotting Average Execution Times

plt.close()
bpl = BarPlot(h*ratio,h)

dic = {
    'Average': [],
    'Variance': [],
}

i = -1
for f in funcs:
    for t in thread_count: 
        tf =  f + "-" + t
        if tf in exceptions:
            continue
        i += 1
 
        dic['Average'].append(  insights[f][t]['per_inv']['exec_time_avg'] )
        dic['Variance'].append( insights[f][t]['per_inv']['exec_time_var'] )

bpl.write_bar( groups, 'Execution Time (seconds) Per Invocation', dic )
bpl.save_fig( 'figure_2_exec_time.jpg' )

plt.close()
bpl = BarPlot(h*ratio,h)

#####################################
#Plotting Figure 3 - Energy Per Invocation against number of threads
#
# Line Plot
#
# x - number of threads - 1,4,8 desktop, 1,4,8,16 victor 
# y - energy per invocation 
# label - given function

def plot_figure_3( fetch_item, y_label: str, fig_name: str ):
    # width / height
    ratio = 1
    h = 8

    plt.close()
    lpl = LinePlot(h*ratio,h)

    x = thread_count 

    dic_y = {
    }

    i = -1
    for f in funcs:
        for t in thread_count: 
            tf =  f + "-" + t
            if tf in exceptions:
                continue
            i += 1
            
            if f not in dic_y:
                dic_y[f] = []
            
            dic_y[f].append( fetch_item(insights,f,t) )

    lpl.write_line( x, dic_y, 'Number of client threads', y_label )
    lpl.save_fig( fig_name  )

## Power Source
fetch_item = lambda x,f,t: x[f][t]['per_inv']['src']['eng']['mean']
y_label = 'Energy Per Inv (Joules) - ' + power_source
fig_name = 'figure_3_energy_'+power_source+'.jpg'
plot_figure_3( fetch_item, y_label, fig_name )

fetch_item = lambda x,f,t: x[f][t]['per_inv']['src']['power']['mean']
y_label = 'Power Per Inv (Watts) - ' + power_source
fig_name = 'figure_3_power_'+power_source+'.jpg'
plot_figure_3( fetch_item, y_label, fig_name )

## Rapl
power_source_o = power_source
power_source = 'rapl'

fetch_item = lambda x,f,t: x[f][t]['per_inv']['rapl']['eng']['mean']
y_label = 'Energy Per Inv (Joules) - ' + power_source
fig_name = 'figure_3_energy_'+power_source+'.jpg'
plot_figure_3( fetch_item, y_label, fig_name )

fetch_item = lambda x,f,t: x[f][t]['per_inv']['rapl']['power']['mean']
y_label = 'Power Per Inv (Watts) - ' + power_source
fig_name = 'figure_3_power_'+power_source+'.jpg'
plot_figure_3( fetch_item, y_label, fig_name )

power_source = power_source_o

## Exec Time, Invocations

fetch_item = lambda x,f,t: x[f][t]['per_inv']['exec_time_avg']
y_label = 'Execution Time (seconds) per invocation'
fig_name = 'figure_3_exec_time.jpg'
plot_figure_3( fetch_item, y_label, fig_name )

fetch_item = lambda x,f,t: x[f][t]['total_invocations']
y_label = 'Invocations'
fig_name = 'figure_3_inv.jpg'
plot_figure_3( fetch_item, y_label, fig_name )

#pprint(minsights)
#pprint(insights)
