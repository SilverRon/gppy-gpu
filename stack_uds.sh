#!/bin/bash

SESSION_NAME="uds_stack"

tmux new-session -d -s $SESSION_NAME

for i in $(seq 1 6); do
    tmux new-window -t $SESSION_NAME: -n "script$i" "python stack_images_mod.py"
done

tmux select-window -t $SESSION_NAME:0

tmux attach-session -t $SESSION_NAME

