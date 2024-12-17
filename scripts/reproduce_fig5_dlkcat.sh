#!/bin/bash

# Script to reproduce ML training, prediction, and analysis runs for
# DLKCat models on CatPred-DB datasets for kcat, Km and Ki

# Author: Veda Sheersh Boorla
# Date: 12-09-2024
# Related to Figure 5 of the manuscript

# Exit script on error
set -e

# Save the initial directory as the base directory
BASE_DIR=$(pwd)

# Define absolute paths for log and output directories
LOG_DIR="$BASE_DIR/../results/reproduce_logs"
OUTPUT_DIR="$BASE_DIR/../results/reproduce_results"

# Ensure the log and output directories exist
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Change to the directory where the experiments are executed
cd external/DLKcat/DeeplearningApproach/Code/model/ #|| exit

# Main pipeline
usage() {
  echo "Usage: $0 [training|analysis]"
  exit 1
}

if [ "$#" -ne 1 ]; then
  usage
fi

case $1 in
  training)
    echo "Running pipeline: Training -> Analysis"
    # Call the run_experiments.sh script for training
    echo "Training runs will take several hours to a day depending on your GPU"
    ./run_experiments.sh "$LOG_DIR"
    ;;
  analysis)
    echo "Running pipeline: Analysis"
    # Analyze results for each parameter and save outputs to the results directory
    python analyze_experiments_new.py kcat $LOG_DIR "$OUTPUT_DIR/kcat/kcat_CatPredDB_DLKcat_results.csv" > $LOG_DIR/kcat_CatPredDB_DLkcat_analysis.log
    echo "Saved results to: $OUTPUT_DIR/kcat/kcat_CatPredDB_DLKcat_results.csv"
    python analyze_experiments_new.py km $LOG_DIR "$OUTPUT_DIR/km/km_CatPredDB_DLKcat_results.csv" > $LOG_DIR/km_CatPredDB_DLkcat_analysis.log
    echo "Saved results to: $OUTPUT_DIR/km/km_CatPredDB_DLKcat_results.csv"
    python analyze_experiments_new.py ki $LOG_DIR "$OUTPUT_DIR/ki/ki_CatPredDB_DLKcat_results.csv" > $LOG_DIR/ki_CatPredDB_DLkcat_analysis.log
    echo "Saved results to: $OUTPUT_DIR/ki/ki_CatPredDB_DLKcat_results.csv"
    ;;
  *)
    usage
    ;;
esac