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

if true; then
    python3 $base/faasmeter/plotting/standalone/native_vs_container_refined.py  \
            -c \
                /data2/ar/iluvatar-energy-experiments/results/trace/desktop/mc_2f_traces_pj/mc_a/functions/fcfs/12/12 \
                /data2/ar/iluvatar-energy-experiments/results/trace/fig_9/desktop/mc_4f_traces_ddp_15min/mc_a/functions/fcfs/12/12 \
            -n \
                /data2/ar/iluvatar-energy-experiments/scripts/experiments/desktop/native_func_stuff/results/native_2funcs_small \
                /data2/ar/iluvatar-energy-experiments/scripts/experiments/desktop/native_func_stuff/results/native_4funcs_5_correct_4_concur
fi

