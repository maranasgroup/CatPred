#!/bin/bash

# Script to reproduce ML training, prediction, and analysis runs for
# DLKCat models on CatPred-DB datasets for kcat, Km and Ki

# Author: Veda Sheersh Boorla
# Date: 12-09-2024
# Related to Figure 5 of the manuscript

# Exit script on error
set -e

cd external/DLKcat/DeeplearningApproach/Code/model/

echo "Training.."

# ./run_experiments.sh

echo "Evaluating.."

python analyze_experiments_new.py kcat
python analyze_experiments_new.py km
python analyze_experiments_new.py ki

