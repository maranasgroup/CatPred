#!/bin/bash

#12 mins per seed
# Array of parameters
params=("km" "ki" "kcat")

# Loop through each parameter
for param in "${params[@]}"; do
    # Loop through seeds 0 to 0 (only one iteration)
    for seed in {0..9}; do
        for dim in 10; do
            python run_model_mpi.py $param --seed $seed --dim $dim > "${param}_${seed}_dim${dim}.log"
        done
    done
done

