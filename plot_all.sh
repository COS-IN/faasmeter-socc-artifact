#!/bin/bash
set -ex

scripts_dir="/extra/alfuerst/repos/faasmeter-data/plotting/execution_scripts/"

pushd $scripts_dir
. ./fig_9.sh &> fig_9.log

. ./neighbor_effect.sh &> neighbor_effect.log

. ./jpt_cdf_single.sh &> jpt_cdf_single.log

. ./error_all_exps.sh &> error_all_exps.log
popd
