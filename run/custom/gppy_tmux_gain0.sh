#!/bin/bash

# Define the observations
observations=(7DT01 7DT02 7DT03 7DT04 7DT05 7DT06 7DT07 7DT08 7DT09 7DT10 7DT11)

# Create a new tmux session named 'gppy0'
tmux new-session -d -s gppy0

# Loop through each observation and create a new tmux window for it
for obs in "${observations[@]}"; do
  tmux new-window -t gppy0 -n "$obs" "python gpwatch_7DT_gain0.py $obs; bash"
done

# Attach to the tmux session
tmux attach-session -t gppy0

