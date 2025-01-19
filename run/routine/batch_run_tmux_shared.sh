#!/bin/bash

# Number of tmux sessions to create
NUM_SESSIONS=15

# Command template
COMMAND_TEMPLATE="python gpwatch_7DT_gain2750.py"

# Custom socket for tmux (if needed for shared sessions)
SOCKET_PATH="/data0/shared_tmux/shared_socket"

# Create the tmux socket directory if it doesn't exist
mkdir -p "$(dirname "$SOCKET_PATH")"

# Loop to create tmux sessions
for i in $(seq 1 $NUM_SESSIONS); do
    SESSION_NAME=$(printf "7DT%02d" $i)
    CMD="$COMMAND_TEMPLATE 7DT$(printf "%02d" $i)"
    
    # Create a new tmux session with the specific command
    tmux -S "$SOCKET_PATH" new-session -d -s "$SESSION_NAME" "$CMD"
    
    echo "Created tmux session: $SESSION_NAME with command: $CMD"
done

# List the created tmux sessions
echo "All tmux sessions:"
tmux -S "$SOCKET_PATH" list-sessions
