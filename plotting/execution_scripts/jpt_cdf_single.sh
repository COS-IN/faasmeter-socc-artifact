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

base=$(realpath $script_path/../../)

python3 $base/plotting/jpt_cdf_single.py  \
        --dirs_desktop \
        $base/results/trace/fig_9/desktop/mc_4f_traces_nddp_15min/mc_*/functions/fcfs/12/12 \
        $base/results/trace/fig_9/desktop/mc_4f_traces_ddp_15min/mc_*/functions/fcfs/12/12 \
        --dirs_server \
        $base/results/trace/fig_9/victor/mc_4f_traces_nddp_15min/mc_*/functions/fcfs/24/24 \
        $base/results/trace/fig_9/victor/mc_4f_traces_ddp_15min/mc_*/functions/fcfs/24/24 \
        --dirs_jetson \
        $base/results/trace/fig_9/jetson/mc_4f_traces_nddp_15min/mc_*/functions/fcfs/12/12 \
        $base/results/trace/fig_9/jetson/mc_4f_traces_ddpm_15min/mc_*/functions/fcfs/12/12 \
        --platform All
