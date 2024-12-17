#!/bin/bash

# Script to reproduce training, prediction, and analysis runs for
# CatPred, UniKP, DLKcat, Baseline models on CatPred-DB datasets for kcat, Km and Ki
# Makes Figure 5 of the manuscript

# Because training all models takes several days, pretrained checkpoints are provided which can be used for prediction directly
# Prediction csv files are also provided which can be used for analysis directly
# So, the script can be run for either training, prediction or analysis
# Training can take total of 10 - 20 days depending on your GPU
# Prediction can take upto few hours 
# Analysis takes a couple of minutes

# Author: Veda Sheersh Boorla
# Date: 12-09-2024

# Exit script on error
set -e

run_training() {
    ./scripts/reproduce_fig5_catpred.sh training
    ./scripts/reproduce_fig5_unikp.sh training
    ./scripts/reproduce_fig5_dlkcat.sh training
    ./scripts/reproducee_fig5_baseline.sh
}

run_prediction() {
    ./scripts/reproduce_fig5_catpred.sh prediction
    ./scripts/reproduce_fig5_unikp.sh prediction
    ./scripts/reproduce_fig5_dlkcat.sh prediction
    ./scripts/reproducee_fig5_baseline.sh
}

# Main pipeline
usage() {
  echo "Usage: $0 [training|prediction|analysis]"
  exit 1
}

if [ "$#" -ne 1 ]; then
  usage
fi

case $1 in
  training)
    echo "Running: Training -> Prediction -> Analysis"
    echo "Note: This will take several days if running on a single GPU. Skip to prediction if you want to start from pretrained models instead."
    run_training
    run_prediction
    run_analysis
    ;;
  prediction)
    echo "Running: Prediction -> Analysis"
    echo "Note: This will take a few hours"
    run_prediction
    run_analysis
    ;;
  analysis)
    echo "Running: Analysis"
    run_analysis
    ;;
  *)
    usage
    ;;
esac

echo "Pipeline starting from $1 completed successfully!"
