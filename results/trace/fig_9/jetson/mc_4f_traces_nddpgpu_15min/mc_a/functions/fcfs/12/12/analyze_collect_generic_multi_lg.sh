#!/bin/bash

IFS=/ read -a wds < <(pwd)
for w in "${wds[@]}"; do 
  if [[ "$w" == "victor" ]]; then 
    platform=server
    cores=24
    break
  elif [[ "$w" == "desktop" ]]; then
    platform=desktop
    cores=12
    break
  elif [[ "$w" == "jetson" ]]; then
    platform=jetson
    cores=12
    break
  fi
done

# single analysis 
if true; then 
  python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                  -l ./    \
                                  -m "$platform"    \
                                  -q quirk_trim_end_by_time    \
                                  -d 0.1  && \
  python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                  -l ./   \
                                  -m "$platform"    \
                                  -p
# no scaphandra 
elif true; then 

  python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                  -s mc_*/functions/fcfs/$cores/$cores/    \
                                  -m "$platform"    \
                                  -q quirk_trim_end_by_time    \
                                  -d 2.1 &&    \
  python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                  -s mc_*/functions/fcfs/$cores/$cores/    \
                                  -m "$platform"    \
                                  -p
# everything 
elif true; then 

  python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                  -s mc_*/functions/fcfs/$cores/$cores/    \
                                  -m "$platform"    \
                                  -q quirk_trim_end_by_time    \
                                  -d 2.1 &&    \
  python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                  -l ./scaphandre/mc_4f_traces_nddp_15min/mc_a/functions/fcfs/$cores/$cores/    \
                                  -m "$platform"    \
                                  -q quirk_trim_end_by_time    \
                                  -d 2.1 &&    \
  python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                  --lg_log_dirs \
                                      ./mc_*/functions/fcfs/$cores/$cores/    \
                                      ./scaphandre/mc_4f_traces_nddp_15min/mc_*/functions/fcfs/$cores/$cores/    \
                                  --mc_type "$platform" \
                                  --collection_only && \
  python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                  -s mc_*/functions/fcfs/$cores/$cores/    \
                                  -m "$platform"    \
                                  -p
# no analysis just collection and plotting 
else
  python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                  -l ./scaphandre/mc_4f_traces_nddp_15min/mc_a/functions/fcfs/$cores/$cores/    \
                                  -m "$platform"    \
                                  -q quirk_trim_end_by_time    \
                                  --no_singular  \
                                  --no_parsing  \
                                  -d 2.1 &&    \
  python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                  -s mc_*/functions/fcfs/$cores/$cores/    \
                                  -m "$platform"    \
                                  -q quirk_trim_end_by_time    \
                                  --no_singular  \
                                  --no_parsing  \
                                  -d 2.1 &&    \
  python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                  --lg_log_dirs \
                                      ./mc_*/functions/fcfs/$cores/$cores/    \
                                      ./scaphandre/mc_4f_traces_nddp_15min/mc_*/functions/fcfs/$cores/$cores/    \
                                  --mc_type "$platform" \
                                  --collection_only && \
  python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                  -s mc_*/functions/fcfs/$cores/$cores/    \
                                  -m "$platform"    \
                                  --plots_only
fi

                                #--plots_only  \
                                #--no_singular  \
                                #--no_parsing  \

