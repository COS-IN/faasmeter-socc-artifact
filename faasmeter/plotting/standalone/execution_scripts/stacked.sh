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
    python3 $base/faasmeter/plotting/standalone/stacked.py  \
            --dirs \
            /data2/ar/iluvatar-energy-experiments/results/trace/desktop/mc_4f_traces_ddp_30min_more_funcs/mc_a_burst/functions/fcfs/12/12 \
            /data2/ar/iluvatar-energy-experiments/results/trace/desktop/mc_4f_traces_ddp_30min_bursty/mc_a/functions/fcfs/12/12
    
    exit 0
fi

