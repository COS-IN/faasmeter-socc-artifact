
o_type='indiv'
o_type='full'

python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                -l ./ \
                                -m server    \
                                -o $o_type    \
                                -q quirk_trim_end_by_time    \
                                -d 0.1 &&    \
python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                -s mc_*/functions/fcfs/24/24/    \
                                -m server    \
                                -o $o_type    \
                                -p

if false; then
    python3 /data2/ar/faasmeter/faasmeter/main.py    \
                                    -l ./ \
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


                                    # -s mc_*/functions/fcfs/24/24/    \
