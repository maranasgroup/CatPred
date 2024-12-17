#!/bin/bash

# Script to reproduce training, prediction, and analysis runs for
# CatPred models on CatPred-DB datasets for kcat, Km and Ki
# Related to Figure 5 of the manuscript

# Author: Veda Sheersh Boorla
# Date: 12-09-2024

# Exit script on error
set -e

# Define variables for directories and files
DATA_DIR="../data/CatPred-DB/"
CKPT_DIR="../data/pretrained/reproduce_checkpoints/"
LOG_DIR="../results/reproduce_logs"
OUTPUT_DIR="../results/reproduce_results"
TRAINING_SCRIPT="train.py"
PREDICTION_SCRIPT="predict.py"
ANALYSIS_SCRIPT="./scripts/analyze_reproduce.py"

# Ensure directories exist
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$OUTPUT_DIR"

# Define functions
run_training() {
  for parameter in ki kcat km; do
  
    if [ "$parameter" == "kcat" ]; then
    echo "Estimated run time for kcat: ~5 h per seed, total ~50 h on NVIDIA-Ampere GPU (Tested on A100)"
    fi

    if [ "$parameter" == "km" ]; then
        echo "Estimated run time for km: ~15 h per seed, total ~150 h on NVIDIA-Ampere GPU (Tested on A100)"
    fi
    
    if [ "$parameter" == "ki" ]; then
        echo "Estimated run time for ki: ~1.7 h per seed, total ~17 h on NVIDIA-Ampere GPU (Tested on A100)"
    fi

    for seed in {0..90..10}; do
      smiles_col="reactant_smiles"
      records_file="./$DATA_DIR/data/$parameter/${parameter}_max_wt_singleSeqs_wpdbs_pdbrecords.json.gz"
      target_col="log10kcat_max"
      if [ "$parameter" != "kcat" ]; then
        smiles_col="substrate_smiles"
        target_col="log10${parameter}_mean"
        records_file="./$DATA_DIR/data/$parameter/${parameter}_mean_wt_singleSeqs_wpdbs_pdbrecords.json.gz" 
      fi

      if [ "$parameter" = "ki" ]; then
        additional_args="--add_pretrained_egnn_feats --pretrained_egnn_feats_path $DATA_DIR/catpred_progres_embeds_dict.pt"
      else
        additional_args=" --add_esm_feats"
      fi

      echo "Starting training for parameter=$parameter and seed=$seed..."
      python "$TRAINING_SCRIPT" \
        --protein_records_path "$records_file" \
        --data_path "./$DATA_DIR/data/$parameter/${parameter}-random_trainval.csv" \
        --dataset_type regression \
        --separate_test_path "./$DATA_DIR/data/$parameter/${parameter}-random_test.csv" \
        --separate_val_path "./$DATA_DIR/data/$parameter/${parameter}-random_val.csv" \
        --smiles_columns "$smiles_col" \
        --target_columns "$target_col" \
        --extra_metrics mae mse r2 \
        --ensemble_size 10 --seq_embed_dim 36 --seq_self_attn_nheads 6 --loss_function mve --batch_size 16 \
        --save_dir "$CKPT_DIR/${parameter}/seed${seed}" --epochs 30\
        $additional_args \
        # > "$LOG_DIR/${parameter}_training_trainvalModel_seed${seed}.log" 2>&1
      echo "Prediction completed for parameter=$parameter and seed=$seed. Logs saved to $LOG_DIR/${parameter}_test_preds_trainvalModel_seed${seed}.log"
    done
  done
}

run_prediction() {
  for parameter in ki km kcat; do
    echo "Estimated run time: ~5 mins per seed, total ~50 mins on NVIDIA-Ampere GPU (Tested on A100)"
    
    for seed in {0..90..10}; do
      smiles_col="reactant_smiles"
      records_file="./$DATA_DIR/data/$parameter/${parameter}_max_wt_singleSeqs_wpdbs_pdbrecords.json.gz"
      if [ "$parameter" != "kcat" ]; then
        smiles_col="substrate_smiles"
        records_file="./$DATA_DIR/data/$parameter/${parameter}_mean_wt_singleSeqs_wpdbs_pdbrecords.json.gz" 
      fi

      if [ "$parameter" = "ki" ]; then
        additional_args="--pretrained_egnn_feats_path $DATA_DIR/catpred_progres_embeds_dict.pt"
      else
        additional_args="--individual_ensemble_predictions"
      fi
      
      echo "Starting prediction for parameter=$parameter and seed=$seed..."
      python "$PREDICTION_SCRIPT" \
        --protein_records_path "$records_file" \
        --test_path "./$DATA_DIR/data/$parameter/${parameter}-random_test.csv" \
        --smiles_columns "$smiles_col" \
        --preds_path "$OUTPUT_DIR/$parameter/${parameter}_test_preds_trainvalModel_seed${seed}.csv" \
        --checkpoint_dir "$CKPT_DIR/${parameter}/seed${seed}" \
        --individual_ensemble_predictions \
        --batch_size 4 \
        $additional_args \
        # > "$LOG_DIR/${parameter}_prediction_trainvalModel_seed${seed}.log" 2>&1
      echo "Prediction completed for parameter=$parameter and seed=$seed. Logs saved to $LOG_DIR/${parameter}_test_preds_trainvalModel_seed${seed}.log"
    done
  done
}

run_analysis() {
  for parameter in kcat km ki; do
    echo ">> Starting analysis for $parameter ..."
    python "$ANALYSIS_SCRIPT" \
        "$parameter" \
        "./$DATA_DIR/data/$parameter/${parameter}-random_trainval.csv" \
        "./$DATA_DIR/data/$parameter/${parameter}-random_test.csv" \
        "$OUTPUT_DIR/${parameter}_analysis-summary.csv" \
        $OUTPUT_DIR/$parameter/${parameter}_test_preds_trainvalModel_seed* \
        > "$LOG_DIR/${parameter}_analysis_trainvalModel.log" 2>&1
    echo "   Analysis completed. Results saved to $OUTPUT_DIR/${parameter}_analysis-summary.csv"
    echo "   Logs saved to $LOG_DIR/${parameter}_test_trainvalModel_analysis.log"
  done
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
