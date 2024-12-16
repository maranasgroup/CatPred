#!/bin/bash

# Script to reproduce ML training, prediction, and analysis runs for
# UniKP models on CatPred-DB datasets for kcat, Km and Ki

# Author: Veda Sheersh Boorla
# Date: 12-09-2024
# Related to Figure 5 of the manuscript

# Exit script on error
set -e

python ./external/UniKP/UniKP_Kcat_v2.py > ./external/UniKP_kcat_results_fig5.txt
python ./external/UniKP/UniKP_Km_v2.py > ./external/UniKP_Km_results_fig5.txt
python ./external/UniKP/UniKP_Ki_v2.py > ./external/UniKP_Km_results_fig5.txt
