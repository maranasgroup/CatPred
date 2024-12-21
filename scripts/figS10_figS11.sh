#!/bin/bash

# Script to reproduce training, prediction, and analysis runs for
# CatPred and UniKP models on DLKcat's dataset for kcat and Kroll's dataset for Km
# Makes Figure S10 and S11 of the manuscript

# Because training all models takes several days, pretrained checkpoints are provided which can be used for prediction directly
# Prediction csv files are also provided which can be used for analysis directly
# So, the script can be run for either training, prediction or analysis
# Training can take total of 4-5 days depending on your GPU
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
    ./scripts/reproduce_figS10_catpred.sh training
    ./scripts/reproduce_figS11_catpred.sh training
    ./scripts/reproduce_figS10_unikp.sh training
    ./scripts/reproduce_figS11_unikp.sh training

    python ./scripts/plot_figS10_figS11.py
    ;;
  prediction)
    echo "Running pipeline: Prediction -> Analysis"
    ./scripts/reproduce_figS10_catpred.sh prediction
    ./scripts/reproduce_figS11_catpred.sh prediction
    ./scripts/reproduce_figS10_unikp.sh analysis
    ./scripts/reproduce_figS11_unikp.sh analysis
    
    python ./scripts/plot_figS10_figS11.py
    ;;
  analysis)
    echo "Running pipeline: Analysis"
    ./scripts/reproduce_figS10_catpred.sh analysis
    ./scripts/reproduce_figS11_catpred.sh analysis
    ./scripts/reproduce_figS10_unikp.sh analysis
    ./scripts/reproduce_figS11_unikp.sh analysis
    
    python ./scripts/plot_figS10_figS11.py
    ;;
  *)
    usage
    ;;
esac

echo "Pipeline starting from $1 completed successfully!"