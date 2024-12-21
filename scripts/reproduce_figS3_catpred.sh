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
LOG_DIR="../data/results/reproduce_logs"
OUTPUT_DIR="../data/results/reproduce_results"
TRAINING_SCRIPT="train.py"
PREDICTION_SCRIPT="predict.py"
ANALYSIS_SCRIPT="./scripts/analyze_reproduce.py"

# Ensure directories exist
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$OUTPUT_DIR"

# Define functions
run_training() {
    echo "Estimated run time for ki: ~1.7 h per seed, total ~17 h on NVIDIA-Ampere GPU (Tested on A100)"
    parameter="ki"
    smiles_col="substrate_smiles"
    target_col="log10${parameter}_mean"
    records_file="./$DATA_DIR/data/$parameter/${parameter}_mean_wt_singleSeqs_wpdbs_pdbrecords.json.gz" 

  for exp in seqemb36_attn6_ens10 seqemb36_attn6_esm_ens10 seqemb36_attn6_ens10_Pretrained_egnnFeats seqemb36_attn6_esm_ens10_Pretrained_egnnFeats; do
    for seed in {0..90..10}; do
        if [[ "$exp" == *"egnn"* ]]; then
            additional_args="--add_pretrained_egnn_feats --pretrained_egnn_feats_path $DATA_DIR/catpred_progres_embeds_dict.pt"
        elif [[ "$exp" == *"esm"* ]]; then
            additional_args="--add_esm_feats"
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
        --save_dir "$CKPT_DIR/${parameter}_ablation_egnn_retrain/$exp/seed${seed}" --epochs 30 \
        $additional_args \
        > "$LOG_DIR/${parameter}_${exp}_training_trainvalModel_seed${seed}.log" 2>&1
      echo "Prediction completed for exp=$exp, parameter=$parameter and seed=$seed. Logs saved to $LOG_DIR/${parameter}_${exp}_test_preds_trainvalModel_seed${seed}.log"
    done
  done
}

run_prediction() {
  local retrain="$1"
  parameter=ki
  for exp in seqemb36_attn6_ens10 seqemb36_attn6_esm_ens10 seqemb36_attn6_ens10_Pretrained_egnnFeats seqemb36_attn6_esm_ens10_Pretrained_egnnFeats; do    
    for seed in {0..90..10}; do
      smiles_col="substrate_smiles"
      records_file="./$DATA_DIR/data/$parameter/${parameter}_mean_wt_singleSeqs_wpdbs_pdbrecords.json.gz"
      if [[ "$exp" == *"egnn"* ]]; then
            additional_args="--add_pretrained_egnn_feats --pretrained_egnn_feats_path $DATA_DIR/catpred_progres_embeds_dict.pt"
      fi
      
      echo "Starting prediction for parameter=$parameter and seed=$seed..."
      python "$PREDICTION_SCRIPT" \
        --protein_records_path "$records_file" \
        --test_path "./$DATA_DIR/data/$parameter/${parameter}-random_test.csv" \
        --smiles_columns "$smiles_col" \
        --preds_path "$OUTPUT_DIR/${parameter}_ablation_egnn$retrain/$exp/${parameter}_test_preds_trainvalModel_seed${seed}.csv" \
        --checkpoint_dir "$CKPT_DIR/${parameter}_ablation_egnn$retrain/$exp/seed${seed}" \
        --individual_ensemble_predictions \
        --batch_size 16 \
        $additional_args \
        # > "$LOG_DIR/${parameter}_${exp}_prediction_trainvalModel_seed${seed}.log" 2>&1
      echo "Prediction completed for exp=$exp, parameter=$parameter and seed=$seed. Logs saved to $LOG_DIR/${parameter}_${exp}_test_preds_trainvalModel_seed${seed}.log"
    done
  done
}

run_analysis() {
    local retrain="$1"
    parameter=ki
  for exp in seqemb36_attn6_ens10 seqemb36_attn6_esm_ens10 seqemb36_attn6_ens10_Pretrained_egnnFeats seqemb36_attn6_esm_ens10_Pretrained_egnnFeats; do
    echo ">> Starting analysis for $parameter ..."
    python "$ANALYSIS_SCRIPT" \
        "$parameter" \
        "./$DATA_DIR/data/$parameter/${parameter}-random_trainval.csv" \
        "./$DATA_DIR/data/$parameter/${parameter}-random_test.csv" \
        "$OUTPUT_DIR/$parameter/${parameter}_${exp}_ablation_egnn_${retrain}_CatPredDB_CatPred_results.csv" \
        $OUTPUT_DIR/${parameter}_ablation_egnn$retrain/$exp/${parameter}_test_preds_trainvalModel_seed* \
        > "$LOG_DIR/${parameter}_analysis_trainvalModel.log" 2>&1
    echo "   Analysis completed. Results saved to $OUTPUT_DIR/${parameter}_${exp}_ablation_egnn_CatPredDB_CatPred_results.csv"
    echo "   Logs saved to $LOG_DIR/${parameter}_${exp}_test_trainvalModel_analysis.log"
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
    run_training _egnn_retrain
    run_prediction _egnn_retrain
    run_analysis
    ;;
  prediction)
    echo "Running pipeline: Prediction -> Analysis"
    run_prediction /
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
