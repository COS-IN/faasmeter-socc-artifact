python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                -s mc_*/functions/fcfs/24/24/    \
                                -m server    \
                                -q quirk_trim_end_by_time    \
                                -d 1.1 \
                                &&    \
python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                -s mc_*/functions/fcfs/24/24/    \
                                -m server    \
                                -p
                                # --no_singular \

if false; then
    python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                    -s mc_*/functions/fcfs/24/24/    \
                                    -m server    \
                                    -o indiv    \
                                    -q quirk_trim_end_by_time    \
                                    -d 2.1 &&    \
    python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                    -s mc_*/functions/fcfs/24/24/    \
                                    -m server    \
                                    -o indiv    \
                                    -p
fi
