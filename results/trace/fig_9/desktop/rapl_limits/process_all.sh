#!/bin/bash 

limits=( 20 29 38 47 65 )

cur_dir=`pwd`
for l in ${limits[@]};do 
    cd "$l/mc_4f_traces_ddp/"
    ./analyze.sh
    cd "$cur_dir"
done


