#!/bin/bash
set -ex

scripts_dir="/extra/alfuerst/repos/faasmeter-data/plotting/execution_scripts/"
venv="plot_venv"
# python3 -m venv $venv/
# source $venv/bin/activate
# python3 -m pip install -r reqs.txt

pushd $scripts_dir
. ./fig_2.sh &> fig_2.log

# . ./fig_6.sh &> fig_6.log

# . ./fig_9.sh &> fig_9.log

# . ./neighbor_effect.sh &> neighbor_effect.log

# . ./jpt_cdf_single.sh &> jpt_cdf_single.log

# . ./error_all_exps.sh &> error_all_exps.log
popd

# deactivate