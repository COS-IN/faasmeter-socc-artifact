# Accountable Carbon Footprints and Energy Profiling For Serverless Functions

Artifacts and results for the paper [Accountable Carbon Footprints and Energy Profiling For Serverless Functions](todo)

## Serverless Control Plane

These were captured and generated using the [Ilúvatar](https://github.com/COS-IN/iluvatar-faas) serverless control plane.
The main driver which captures energy and performance details are collected is in [this set of files](https://github.com/COS-IN/iluvatar-faas/tree/master/src/Il%C3%BAvatar/iluvatar_library/src/energy).
For example it can pull data from [Intel and AMD RAPL registers](https://github.com/COS-IN/iluvatar-faas/blob/master/src/Il%C3%BAvatar/iluvatar_library/src/energy/rapl.rs) and query [baseboard management controller IPMI data](https://github.com/COS-IN/iluvatar-faas/blob/master/src/Il%C3%BAvatar/iluvatar_library/src/energy/ipmi.rs).
These metrics are collected and fed into the [FaasMeter system](./faasmeter/) to determine per-function and per-invocation energy usage and carbon fotprints.
Our control plane also possesses the capability to [limit dispatches of enqueued invocations](https://github.com/COS-IN/iluvatar-faas/blob/master/src/Il%C3%BAvatar/iluvatar_worker_library/src/services/invocation/energy_limiter.rs) if the energy consumption of the hardware is too high.


## Prerequisites

To run the analysis and graphing scripts, the following dependencies are needed.

```sh
sudo apt install python3 python3-pip
```

Then run this script which will generate all the plots.
```sh
./plot_all.sh
```

## Generated Figures

These are all the plots from the paper that are generated by the artifacts and scripts in this repository.
Traces can be found in the output-full-mc_a.json file in the corresponding linked directories. 

Figure 2
![Fig 2](./plotting/execution_scripts/legend.jpg)
![Fig 2a](./plotting/execution_scripts/fig_2_smart_small_funcs.jpg)
![Fig 2b](./plotting/execution_scripts/fig_2_smart_large_funcs.jpg)


Figure 6
![Fig 6](./results/trace/desktop/mc_4f_30min_traces/mc_a/fcfs/12/12/kf-stab.png)

Figure 8
![Fig 8](./results/trace/fig_9/desktop/mc_4f_traces_nddp_15min/mc_a/functions/fcfs/12/12/dfs/plots/standalone/fig_9.png)

![Trace for Figure 8](./results/trace/for_paper/victor/nddp/mc_4f_traces_nddp_15min/mc_a/functions/fcfs/24/24/output-full-mc_a.json)

Figure 10a
![Fig 10a](./results/trace/desktop/mc_4f_traces_ddp_30min_bursty/mc_a/functions/fcfs/12/12/dfs/plots/standalone/stacked_f23.png)

Figure 10b
![Fig 10b](./results/trace/desktop/mc_4f_traces_ddp_30min_more_funcs/mc_a_burst/functions/fcfs/12/12/dfs/plots/standalone/stacked_f23.png)

![Trace for Figure 10b](./results/trace/desktop/mc_4f_traces_ddp_30min_bursty/mc_a/functions/fcfs/12/12/output-full-mc_a.json)

Figure 10c
![Fig 10c](./plotting/execution_scripts/jpt_error_All.png)

![Trace for Figure 10c](./results/trace/victor/mc_4f_traces_allcpu_30min/mc_a/functions/fcfs/24/24/worker_worker1.log)

Figure 11
![Fig 11](./results/trace/fig_9/desktop/mc_4f_traces_ddp_15min/mc_a/functions/fcfs/12/12/dfs/plots/standalone/neighboreffect_plot.png)

Figure 12
![Fig 12](./results/trace/fig_9/desktop/mc_4f_traces_nddp_15min/mc_a/functions/fcfs/12/12/dfs/plots/standalone/jpt_ratio_cdf_singleAll.png)
