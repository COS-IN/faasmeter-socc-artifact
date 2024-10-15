
base=/data2/ar/iluvatar-energy-experiments

if true; then
    python3 /data2/ar/faasmeter/faasmeter/plotting/standalone/error_perct_vs_std_nofunc.py  \
              --dirs_desktop \
                $base/results/trace/fig_9/desktop/mc_4f_traces_nddp_15min/mc_a/functions/fcfs/12/12/ \
                $base/results/trace/fig_9/desktop/mc_4f_traces_ddp_15min/mc_a/functions/fcfs/12/12/ \
              --dirs_server \
                $base/results/trace/fig_9/victor/mc_4f_traces_nddp_15min/mc_a/functions/fcfs/24/24/   \
                $base/results/trace/fig_9/victor/mc_4f_traces_ddp_15min/mc_a/functions/fcfs/24/24/   \
              --save_plot \
                $base/results/trace/fig_9/desktop/mc_4f_traces_nddp_15min/mc_a/functions/fcfs/12/12/ \
                && true
fi

if false; then
    python3 /data2/ar/faasmeter/faasmeter/plotting/standalone/error_perct_vs_std.py -s \
                $base/results/trace/fig_9/desktop/mc_4f_traces_ddp_15min/mc_a/functions/fcfs/12/12/ \
                $base/results/trace/fig_9/victor/mc_4f_traces_ddp_15min/mc_a/functions/fcfs/24/24/   \
                && true
fi
