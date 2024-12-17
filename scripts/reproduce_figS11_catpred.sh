#!/bin/bash

# Script to reproduce training, prediction, and analysis runs for
# CatPred models on Kroll et. al. Km dataset
# Related to Figure S11 of the manuscript

# Author: Veda Sheersh Boorla
# Date: 12-09-2024

# Exit script on error
set -e

# Define variables for directories and files
DATA_DIR="./CatPred-DB/"
CKPT_DIR="./pretrained/reproduce_checkpoints/"
LOG_DIR="./output/reproduce_logs"
OUTPUT_DIR="./output/reproduce_results"
TRAINING_SCRIPT="train.py"
PREDICTION_SCRIPT="predict.py"
ANALYSIS_SCRIPT="./scripts/analyze_reproduce.py"

# Ensure directories exist
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$OUTPUT_DIR"

# Define functions
run_training() {
    echo "Estimated run time: ~3 h per seed, total ~15 h on NVIDIA-Ampere GPU (Tested on A100)"
    for seed in {0..40..10}; do
      smiles_col="substrate_smiles"
      records_file="./CatPred-DB/external_data/Km_test_11722_pdbrecords.json"
      target_col="log10km_mean"

      echo "Starting training for parameter=kcat on DLKCat dataset and seed=$seed..."
      python "$TRAINING_SCRIPT" \
        --protein_records_path "$records_file" \
        --data_path "./CatPred-DB/external_data/Km_test_11722_train.csv" \
        --dataset_type regression \
        --separate_test_path "./CatPred-DB/external_data/Km_test_11722_test.csv" \
        --separate_val_path "./CatPred-DB/external_data/Km_test_11722_val.csv" \
        --smiles_columns "$smiles_col" \
        --target_columns "$target_col" \
        --extra_metrics mae mse r2 --add_esm_feats \
        --ensemble_size 10 --seq_embed_dim 36 --seq_self_attn_nheads 6 --loss_function mve --batch_size 32 \
        --save_dir "$CKPT_DIR/km_kroll/seed${seed}" --epochs 30 \
        # > "$LOG_DIR/kcat_training_DLKcatDatasetModel_seed${seed}.log" 2>&1
      echo "Prediction completed for parameter=km and seed=$seed. Logs saved to $LOG_DIR/km_training_KrollDatasetModel_seed${seed}.log"
    done
}

run_prediction() {
    echo "Estimated run time: ~2-3 mins per seed, total ~15 mins on NVIDIA-Ampere GPU (Tested on A100)"
    for seed in {0..40..10}; do
      smiles_col="substrate_smiles"
      records_file="./CatPred-DB/external_data/Km_test_11722_pdbrecords.json"
      target_col="log10km_mean"
      
      echo "Starting prediction for parameter=kcat on DLKCat dataset and seed=$seed..."
      python "$PREDICTION_SCRIPT" \
        --protein_records_path "$records_file" \
        --test_path "./CatPred-DB/external_data/Km_test_11722_test.csv" \
        --smiles_columns "$smiles_col" \
        --preds_path "$OUTPUT_DIR/km_kroll/km_kroll_preds_KrollDatasetModel_seed${seed}.csv" \
        --checkpoint_dir "$CKPT_DIR/km_kroll/seed${seed}" \
        --individual_ensemble_predictions \
        $additional_args \
        # > "$LOG_DIR/km_kroll_preds_KrollDatasetModel_seed${seed}.log" 2>&1
      echo "Prediction completed for parameter=km and seed=$seed. Logs saved to $LOG_DIR/km_kroll_preds_KrollDatasetModel_seed${seed}.log"
    done
}

run_analysis() {
    echo ">> Starting analysis for km on Kroll dataset ..."
    python "$ANALYSIS_SCRIPT" \
        km \
        "./CatPred-DB/external_data/Km_test_11722_train.csv" \
        "./CatPred-DB/external_data/Km_test_11722_test.csv" \
        "$OUTPUT_DIR/km_kroll_analysis-summary.csv" \
        $OUTPUT_DIR/km_kroll/km_kroll_preds_KrollDatasetModel_seed* \
        > "$LOG_DIR/km_kroll_analysis_trainvalModel.log" 2>&1
    echo "   Analysis completed. Results saved to $OUTPUT_DIR/km_kroll_analysis-summary.csv"
    echo "   Logs saved to $LOG_DIR/km_kroll_analysis_trainvalModel.log"
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
    echo "Running pipeline: Training -> Prediction -> Analysis"
    run_training
    run_prediction
    run_analysis
    ;;
  prediction)
    echo "Running pipeline: Prediction -> Analysis"
    run_prediction
    run_analysis
    ;;
  analysis)
    echo "Running pipeline: Analysis"
    run_analysis
    ;;
  *)
    usage
    ;;
esac

echo "Pipeline starting from $1 completed successfully!"
