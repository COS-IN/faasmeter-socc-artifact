
dir_lg2df = '/dfs'
dir_cpushare = dir_lg2df + '/cpu'
dir_combined = dir_lg2df + '/combined'
dir_analysis = dir_lg2df + '/analysis'
dir_plots = dir_lg2df + '/plots'
dir_plots_stacked = dir_plots + '/stacked' 
dir_plots_mc_lines = dir_plots + '/mc_lines' 
dir_plots_mc_bars = dir_plots + '/mc_bars' 

N_init = 120
N = 60
delta = 1
cpu_threshold = 5.0

o_type = 'indiv' 
o_type = 'full' 

update_type = 'cumulative'
update_type = 'kalman'
kf_type='j'
kf_type='x'
kf_type='x-n' # o_type must be full 

g = globals()
if 'tag_psys' not in g:
    tag_psys = []
ipmi_correction = 120.0  
ipmi_correction = 0.0  

def set_global_mc_type( mc ):
    g = globals()
    g['mc_type'] = mc
    global tag_psys
    if mc_type == 'desktop':
        #g['tag_psys'] = 'igpm' 
        tag_psys.append( 'igpm' )
    elif mc_type == 'server':
        #g['tag_psys'] = 'ipmi' 
        tag_psys.append( 'ipmi' )
    elif mc_type == 'jetson':
        #g['tag_psys'] = 'tegra' 
        tag_psys.append( 'tegra' )

quirks = {
    'quirk_trim_end': False,
    'quirk_trim_end_by_time': False,
}

quirks_data = {
    k: None for k in quirks.keys()
}

dissag_cpu_execptions = {
    'jetson'
}

