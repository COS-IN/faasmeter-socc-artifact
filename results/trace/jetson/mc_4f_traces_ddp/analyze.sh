

if true; then
# if false; then
python3 /data2/ar/faasmeter/faasmeter/main.py   \
                    -l mc_a/functions/fcfs/12/12/   \
                    -m jetson   \
                    -o indiv   \
                    -q quirk_trim_end_by_time   \
                    -d 2.1 && \
python3 /data2/ar/faasmeter/faasmeter/main.py   \
                    -l mc_a/functions/fcfs/12/12/   \
                    -m jetson   \
                    -o indiv   \
                    -p
fi

if false; then
#if true; then
    python3 /data2/ar/faasmeter/faasmeter/main.py   \
                        -s mc_*/functions/fcfs/12/12/   \
                        -m jetson   \
                        -o indiv   \
                        -q quirk_trim_end_by_time   \
                        -d 2.1 && \
    python3 /data2/ar/faasmeter/faasmeter/main.py   \
                        -s mc_*/functions/fcfs/12/12/   \
                        -m jetson   \
                        -o indiv   \
                        -p
fi
