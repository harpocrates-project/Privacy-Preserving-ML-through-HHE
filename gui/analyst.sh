#!/usr/bin/env bash
set -e

# Clean up background monitor on exit
cleanup() {
    kill "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

# Start directory monitor in background
inotifywait -m -e close_write --format '%w%f' . |
    while IFS= read -r file; do
        [[ "$file" == *.txt ]] && gui/utils/plot_binaryoutput.py "$file" &
    done &
monitor_pid=$!

# Run the wrapped executable, forwarding its output to the terminal
./build/analyst 2>&1

# When the executable finishes the trap will terminate the monitor
