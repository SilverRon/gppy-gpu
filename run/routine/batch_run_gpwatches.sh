#!/bin/bash

# ??: ? ?? ?? ? ? ? ? ??? (?)
NUM_REPETITIONS=16
TIME_TO_SLEEP=600

# ?? ??
for i in $(seq 2 $NUM_REPETITIONS); do
    TAB_NAME=$(printf "7DT%02d" $i)
    gnome-terminal --tab --title="$TAB_NAME" -- bash -c "source ~/.bashrc; conda activate pipeline; cd ~/gppy/run/routine; python gpwatch_7DT_gain2750_test.py 7DT$(printf "%02d" $i); exec bash"
    sleep $TIME_TO_SLEEP
done
