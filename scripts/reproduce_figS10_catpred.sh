#!/bin/bash

# Script to reproduce training, prediction, and analysis runs for
# CatPred models on DLKcat kcat dataset
# Related to Figure S10 of the manuscript

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
    echo "Estimated run time: ~4 h per seed, total ~20 h on NVIDIA-Ampere GPU (Tested on A100)"
    for seed in {0..40..10}; do
      smiles_col="substrate_smiles"
      records_file="./CatPred-DB/external_data/Kcat_combination_0918_wildtype_mutant_pdbrecords.json"
      target_col="log10kcat_max"

      echo "Starting training for parameter=kcat on DLKCat dataset and seed=$seed..."
      python "$TRAINING_SCRIPT" \
        --protein_records_path "$records_file" \
        --data_path "./CatPred-DB/external_data/Kcat_combination_0918_wildtype_mutant_train.csv" \
        --dataset_type regression \
        --separate_test_path "./CatPred-DB/external_data/Kcat_combination_0918_wildtype_mutant_test.csv" \
        --separate_val_path "./CatPred-DB/external_data/Kcat_combination_0918_wildtype_mutant_val.csv" \
        --smiles_columns "$smiles_col" \
        --target_columns "$target_col" \
        --extra_metrics mae mse r2 --add_esm_feats \
        --ensemble_size 10 --seq_embed_dim 36 --seq_self_attn_nheads 6 --loss_function mve --batch_size 32 \
        --save_dir "$CKPT_DIR/kcat_dlkcat/seed${seed}" --epochs 30 \
        # > "$LOG_DIR/kcat_training_DLKcatDatasetModel_seed${seed}.log" 2>&1
      echo "Prediction completed for parameter=kcat and seed=$seed. Logs saved to $LOG_DIR/kcat_training_DLKcatDatasetModel_seed${seed}.log"
    done
}

run_prediction() {
    echo "Estimated run time: ~2-3 mins per seed, total ~15 mins on NVIDIA-Ampere GPU (Tested on A100)"
    for seed in {0..40..10}; do
      smiles_col="substrate_smiles"
      records_file="./CatPred-DB/external_data/Kcat_combination_0918_wildtype_mutant_pdbrecords.json"
      target_col="log10kcat_max"
      
      echo "Starting prediction for parameter=kcat on DLKCat dataset and seed=$seed..."
      python "$PREDICTION_SCRIPT" \
        --protein_records_path "$records_file" \
        --test_path "./CatPred-DB/external_data/Kcat_combination_0918_wildtype_mutant_test.csv" \
        --smiles_columns "$smiles_col" \
        --preds_path "$OUTPUT_DIR/kcat_dlkcat/kcat_dlkcat_preds_DLKcatDatasetModel_seed${seed}.csv" \
        --checkpoint_dir "$CKPT_DIR/kcat_dlkcat/seed${seed}" \
        --individual_ensemble_predictions \
        $additional_args > "$LOG_DIR/kcat_prediction_trainvalModel_seed${seed}.log" 2>&1
      echo "Prediction completed for parameter=kcat and seed=$seed. Logs saved to $LOG_DIR/kcat_test_preds_trainvalModel_seed${seed}.log"
    done
}

run_analysis() {
    echo ">> Starting analysis for kcat on DLKCat dataset ..."
    python "$ANALYSIS_SCRIPT" \
        kcat \
        "./CatPred-DB/external_data/Kcat_combination_0918_wildtype_mutant_train.csv" \
        "./CatPred-DB/external_data/Kcat_combination_0918_wildtype_mutant_test.csv" \
        "$OUTPUT_DIR/kcat_dlkcat_analysis-summary.csv" \
        $OUTPUT_DIR/kcat_dlkcat/kcat_dlkcat_preds_DLKcatDatasetModel_seed* \
        > "$LOG_DIR/kcat_dlkcat_analysis_trainvalModel.log" 2>&1
    echo "   Analysis completed. Results saved to $OUTPUT_DIR/kcat_dlkcat_analysis-summary.csv"
    echo "   Logs saved to $LOG_DIR/kcat_dlkcat_analysis_trainvalModel.log"
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
