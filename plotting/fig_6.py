from faasmeter.faasmeter.parsing.Logs_to_df import Logs_to_df, function_name_to_paper
# from Logs_to_df import Logs_to_df, function_name_to_paper
from faasmeter.faasmeter.disaggregation.Kalman_Filter import Kalman_Filter 
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import os.path
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams['axes.linewidth'] = 0.5
matplotlib.rcParams['axes.labelpad'] = 2.0
matplotlib.rcParams['axes.titlepad'] = 2.0
matplotlib.rcParams['figure.dpi'] = 200.0
import copy 

loc = "../../results/trace/desktop/mc_4f_30min_traces/mc_a/fcfs/12/12"
edata = Logs_to_df(loc, "desktop")
edata.output_type = "indiv"
edata.populate_system_settings()
edata.process_all_logs()
#edata.power_src='perf_rapl'

mkf = Kalman_Filter()
mkf.ldf = edata 
delta = 1 
mkf.update_p = False
# Run all the way to the end. 
N_init = 100
N = 60
mkf.kalman_over_time(N_init, N, delta, "kalman")

def kf_for_type(kf_type):
    edata = Logs_to_df(loc, "desktop")
    edata.output_type = "indiv"
    edata.populate_system_settings()
    edata.process_all_logs()

    kf = Kalman_Filter()
    kf.update_type = kf_type
    kf.ldf = edata 
    delta = 1 
    kf.ks_hist = []
    kf.update_p = False
        # Run all the way to the end. 
    N_init = 100
    N = 60
    print(kf.ks_hist)
    kf.kalman_over_time(N_init, N, delta)
    return kf 

def plot_vs_time(kfs):
    fig, axs = plt.subplots(ncols=3)
    fig.set_size_inches(16,3)
    edata = Logs_to_df(loc, "desktop")
    edata.output_type = "indiv"
    edata.populate_system_settings()
    edata.process_all_logs()

    kf_types = ["kalman", "memoryless", "cumulative"]
    
    for ax, kf_type, kf  in zip(axs, kf_types, kfs):
        #kf = kf_for_type(kf_type)
        jdf = kf.ks_over_time("J")
        times = kf.ks_over_time("t")
        rel_times = times - times[0][0]
        times = [x.total_seconds() for x in rel_times[0]]
        print(kf_type)
        for i in list(jdf.columns)[:-1]: #ctrl-plane energy per invok makes no sense 
            ax.plot(jdf[i], label=function_name_to_paper(kf.ldf.princip_list[i]))
            ax.set_title(kf_type)
            ax.set_ylim((0,100))
            if kf_type == "kalman":
                ax.set_ylabel("Per-invocation Energy (J)")
            if kf_type == "memoryless":
                ax.set_xlabel("Time-Steps")
        del kf 
        #ax.legend(ncol)
        
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.57, 1.2), ncol=4)
    name = os.path.join(loc, "kf-stab")
    # plt.savefig(name,bbox_inches='tight')
    plt.savefig( name + '.png', bbox_inches='tight') 
    plt.savefig( name + '.pdf', bbox_inches='tight' ) 
    print('Saved: {}'.format(name))
    plt.close()

kf_types = ["kalman", "memoryless", "cumulative"]

kfs = [kf_for_type(x) for x in kf_types]
plot_vs_time(kfs)