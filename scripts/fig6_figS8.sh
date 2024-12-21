#!/bin/bash

# Script to reproduce the uncertainty analysis runs for CatPred models
# 
# Makes Figures 6 and S8 of the manuscript

# Author: Veda Sheersh Boorla
# Date: 12-09-2024

# Exit script on error
set -e

# Main pipeline
usage() {
  echo "Usage: $0 [prediction|analysis]"
  exit 1
}

if [ "$#" -ne 1 ]; then
  usage
fi

case $1 in
  prediction)
    echo "Running pipeline: Prediction -> Analysis"
    ./scripts/reproduce_uncertainty.sh prediction
    
    python ./scripts/plot_fig6_figS8.py
    ;;
  analysis)
    echo "Running pipeline: Analysis"
    python ./scripts/plot_fig6_figS8.py
    ;;
  *)
    usage
    ;;
esac

echo "Pipeline starting from $1 completed successfully!"