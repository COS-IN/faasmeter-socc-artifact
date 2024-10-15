#!/bin/bash

get_real_path() {
    script_path=$(dirname $1)
    script_path=$(realpath $script_path)
}
if [[ $0 != $BASH_SOURCE ]]; then
    get_real_path $BASH_SOURCE
else
    get_real_path $0
fi
echo $script_path

base=$(realpath $script_path/../../../../)

if false; then 
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/jetson/mc_4f_traces_ddpm_15min/mc_*/functions/fcfs/12/12 \
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/jetson/mc_4f_traces_ddpm_15min/mc_*/functions/fcfs/12/12 \
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/desktop/mc_4f_traces_ddp_15min/mc_*/functions/fcfs/12/12 \
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/desktop/mc_4f_traces_ddp_15min/mc_*/functions/fcfs/12/12 
fi

if true; then
    python3 $base/faasmeter/plotting/standalone/error_all_exps.py -s \
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/desktop/rapl_limits/*/mc_4f_traces_ddp/mc_*/functions/fcfs/12/12 \
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/desktop/mc_4f_traces_nddp_15min/mc_*/functions/fcfs/12/12 \
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/desktop/mc_4f_traces_ddp_15min/mc_*/functions/fcfs/12/12 \
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/victor/mc_4f_traces_nddp_15min/mc_*/functions/fcfs/24/24 \
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/victor/mc_4f_traces_ddp_15min/mc_*/functions/fcfs/24/24 \
            /data2/ar/iluvatar-energy-experiments/results/trace/for_paper/desktop/mc_pyaes_inputsizes/mc_*/functions/fcfs/12/12 \
            /data2/ar/iluvatar-energy-experiments/results/trace/desktop/mc_scaling/mc_4f_traces_ddp*/mc_*/functions/fcfs/12/12 \
            /data2/ar/iluvatar-energy-experiments/results/trace/desktop/mc_4f_traces_ddp_30min_more_funcs/mc_a_burst/functions/fcfs/12/12 \
            --platform All

    exit 0

    python3 $base/faasmeter/plotting/standalone/error_all_exps.py -s \
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/desktop/mc_4f_traces_nddp_15min/mc_*/functions/fcfs/12/12 \
            --platform Desktop 

    python3 $base/faasmeter/plotting/standalone/error_all_exps.py -s \
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/victor/mc_4f_traces_nddp_15min/mc_*/functions/fcfs/24/24 \
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/victor/mc_4f_traces_ddp_15min/mc_*/functions/fcfs/24/24 \
            --platform Server 

    python3 $base/faasmeter/plotting/standalone/error_all_exps.py -s \
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/jetson/mc_4f_traces_nddp_15min/mc_*/functions/fcfs/12/12 \
            --platform Jetson 

#            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/jetson/mc_4f_traces_ddpm_15min/mc_m/functions/fcfs/12/12

    #        /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/victor/mc_4f_traces_ddp_15min/mc_v/functions/fcfs/24/24 \
    #                $base/data/desktop/exp_ddp  \
    #                $base/data/server/exp_ddp   \
    #                $base/data/jetson/exp_ddpm  

fi

if false; then
    python3 $base/faasmeter/plotting/standalone/combined_fig_9_errors.py -s \
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/desktop/mc_4f_traces_ddp_15min/mc_v/functions/fcfs/12/12 \
            /data2/ar/iluvatar-energy-experiments/results/trace/victor/mc_4f_traces_ddp_15min/mc_v/functions/fcfs/24/24 \
            /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/jetson/mc_4f_traces_ddpm_15min/mc_m/functions/fcfs/12/12

    #        /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/victor/mc_4f_traces_ddp_15min/mc_v/functions/fcfs/24/24 \
    #                $base/data/desktop/exp_ddp  \
    #                $base/data/server/exp_ddp   \
    #                $base/data/jetson/exp_ddpm  

fi

if false; then
    python3 /data2/ar/faasmeter/faasmeter/plotting/standalone/combined_fig_9_errors.py -s \
               /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/desktop/mc_4f_traces_ddp_15min/mc_a/functions/fcfs/12/12 \
               /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/victor/mc_4f_traces_ddp_15min/mc_a/functions/fcfs/24/24 \
               /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/jetson/mc_4f_traces_ddpm_15min/mc_a/functions/fcfs/12/12 
fi

