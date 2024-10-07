if true; then
#if false; then
    python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                    -l mc_a/functions/fcfs/12/12/    \
                                    -m desktop    \
                                    -q quirk_trim_end_by_time    \
                                    -d 2.1 &&    \
    python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                    -l mc_a/functions/fcfs/12/12/    \
                                    -m desktop    \
                                    -p
fi



if false; then
#if true; then
    python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                    -s mc_*/functions/fcfs/12/12/    \
                                    -m desktop    \
                                    -o indiv    \
                                    -q quirk_trim_end_by_time    \
                                    -d 0.1 &&    \
    python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                    -s mc_*/functions/fcfs/12/12/    \
                                    -m desktop    \
                                    -o indiv    \
                                    -p
fi

if true; then
#if false; then
    python3 /data2/ar/faasmeter/faasmeter/plotting/standalone/input_sizes.py \
                                    -d mc_a/functions/fcfs/12/12/
fi

# multi only
if false; then
#if true; then
    python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                    -s mc_*/functions/fcfs/12/12/    \
                                    -m desktop    \
                                    -o indiv    \
                                    -q quirk_trim_end_by_time    \
                                    -d 0.1     \
                                    -n
fi
