#!/bin/bash
set -e

base=./trace/desktop
# echo $base
# for sub_dir in $base/*/     # list directories in the form "/tmp/dirname/"
for sub_dir in mc_scaling mc_traces trace3 why_need_it;
do
    echo "$base/$sub_dir"    # print everything after the final "/"
    git add $base/$sub_dir
    git commit -m "$base/$sub_dir data"
    git push
done