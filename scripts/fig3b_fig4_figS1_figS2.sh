#!/bin/bash

# Script to reproduce training, prediction, and analysis runs for
# CatPred, UniKP, DLKcat, Baseline models on CatPred-DB datasets for kcat, Km and Ki
# Makes Figure 5 of the manuscript

# Because training all models takes several days, pretrained checkpoints are provided which can be used for prediction directly
# Prediction csv files are also provided which can be used for analysis directly
# So, the script can be run for either training, prediction or analysis
# Training can take total of 10 days or lesser depending on your GPU
# Prediction can take upto few hours 
# Analysis takes a couple of minutes

# Author: Veda Sheersh Boorla
# Date: 12-09-2024

# Exit script on error
set -e

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
    echo "Running pipeline: Training -> Prediction -> Analysis"
    ./scripts/reproduce_ablation.sh training
    python ./scripts/plot_fig3b_fig4_figS1_figS2.py
    ;;
  prediction)
    echo "Running pipeline: Prediction -> Analysis"
    ./scripts/reproduce_ablation.sh prediction
    python ./scripts/plot_fig3b_fig4_figS1_figS2.py
    ;;
  analysis)
    echo "Running pipeline: Analysis"
    ./scripts/reproduce_ablation.sh analysis
    python ./scripts/plot_fig3b_fig4_figS1_figS2.py
    ;;
  *)
    usage
    ;;
esac

echo "Pipeline starting from $1 completed successfully!"