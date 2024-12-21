#!/bin/bash

# Accept the absolute path to the log directory as an argument
LOG_DIR=$1

# Ensure the log directory exists
mkdir -p "$LOG_DIR"

# Array of parameters to iterate through
PARAMS=("km" "ki" "kcat")

# Loop through each parameter
for param in "${PARAMS[@]}"; do
    # Loop through seeds 0 to 9
    for seed in {0..9}; do
        # Use a fixed dimension (20 in this case)
        for dim in 20; do
            # Define the log file path for each experiment
            LOG_FILE="$LOG_DIR/${param}_${seed}_dim${dim}.log"

            # Run the experiment and save the output in the corresponding log file
            python run_model_mpi.py "$param" --seed "$seed" --dim "$dim" > "$LOG_FILE"
        done
    done
done
