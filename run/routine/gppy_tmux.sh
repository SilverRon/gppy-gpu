#!/bin/bash

# Define the observations
# observations=(7DT01 7DT02 7DT03 7DT04 7DT05 7DT06 7DT07 7DT08 7DT09 7DT10 7DT11 7DT12 7DT13 7DT14 7DT15)
# observations=(7DT01 7DT02 7DT03 7DT04 7DT05 7DT06 7DT07 7DT08 7DT09 7DT10 7DT11 7DT12 7DT13 7DT14 7DT15 7DT16 7DT17 7DT18 7DT19 7DT20)
observations=(7DT01 7DT02 7DT03 7DT04 7DT05 7DT06 7DT07 7DT08 7DT09 7DT10 7DT11 7DT13)

# Create a new tmux session named 'gppy'
tmux new-session -d -s gppy

# Loop through each observation and create a new tmux window for it
for obs in "${observations[@]}"; do
  tmux new-window -t gppy: -n "$obs" "python gpwatch_7DT_gain2750.py $obs; bash"
done

# Attach to the tmux session
tmux attach-session -t gppy
