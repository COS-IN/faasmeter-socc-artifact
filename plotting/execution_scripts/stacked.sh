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

python3 $base/plotting/stacked.py  \
        --dirs \
        $base/results/trace/desktop/mc_4f_traces_ddp_30min_more_funcs/mc_a_burst/functions/fcfs/12/12 \
        $base/results/trace/desktop/mc_4f_traces_ddp_30min_bursty/mc_a/functions/fcfs/12/12
        # $base/results/trace/desktop/mc_4f_traces_ddp_30min_more_funcs/mc_a_burst/functions/fcfs/12/12 \
