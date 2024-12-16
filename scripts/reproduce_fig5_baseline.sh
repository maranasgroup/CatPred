#!/bin/bash

# Script to reproduce ML training, prediction, and analysis runs
# Author: Veda Sheersh Boorla
# Date: 12-09-2024

# Exit script on error
set -e

OUTPUT_DIR="../results/reproduce_results"
LOG_DIR="../results/reproduce_logs"
PARAMS=("kcat" "km" "ki")

for param in "${PARAMS[@]}"; do
  echo "Evaluating $param prediction"
  python ./scripts/baseline_analysis.py $param $OUTPUT_DIR/${param}_baseline_analysis.csv > $LOG_DIR/${param}_baseline_analysis.log
done
