#!/bin/bash

# Script to reproduce ML training, prediction, and analysis runs
# Author: Veda Sheersh Boorla
# Date: 12-09-2024

# Exit script on error
set -e

OUTPUT_DIR="../results/reproduce_results"
LOG_DIR="../results/reproduce_logs"
PARAMS=("kcat" "km" "ki")

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
    echo "Training runs will take several hours"
    for param in "${PARAMS[@]}"; do
      echo "Baseline $param prediction"
      python ./scripts/baseline_analysis.py recalculate $param $OUTPUT_DIR/$param/${param}_CatPredDB_Baseline_results.csv #> $LOG_DIR/${param}_baseline_analysis.log
    done
    ;;
  analysis)
    echo "Running pipeline: Analysis"
    for param in "${PARAMS[@]}"; do
      echo "Baseline $param prediction"
      python ./scripts/baseline_analysis.py analysis $param $OUTPUT_DIR/$param/${param}_CatPredDB_Baseline_results.csv #> $LOG_DIR/${param}_baseline_analysis.log
    done
    ;;
  *)
    usage
    ;;
esac