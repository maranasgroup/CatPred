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
ANALYSIS_SCRIPT="./scripts/uncertainty_analysis.py"

# Ensure directories exist
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$OUTPUT_DIR"

run_prediction() {
    # List of experiments
    local retrain="$1"
    directories=(
        substrate_only
        seqemb36_attn6_ens10
        seqemb36_attn6_esm_ens10
        seqemb36_attn6_esm_ens10_Pretrained_egnnFeats
    )

    for parameter in kcat km ki; do
        if [ "$parameter" = "kcat" ]; then
            smiles_col="reactant_smiles"
            records_file="./$DATA_DIR/data/$parameter/${parameter}_max_wt_singleSeqs_wpdbs_pdbrecords.json.gz"
            additional_args="--uncertainty_method mve"
            dir="seqemb36_attn6_esm_ens10"
        elif [ "$parameter" = "km" ]; then
            smiles_col="substrate_smiles"
            records_file="./$DATA_DIR/data/$parameter/${parameter}_mean_wt_singleSeqs_wpdbs_pdbrecords.json.gz"
            additional_args="--uncertainty_method mve"
            dir="seqemb36_attn6_esm_ens10"
        elif [ "$parameter" = "ki" ]; then
            smiles_col="substrate_smiles"
            records_file="./$DATA_DIR/data/$parameter/${parameter}_mean_wt_singleSeqs_wpdbs_pdbrecords.json.gz"
            dir="seqemb36_attn6_esm_ens10_Pretrained_egnnFeats"
            additional_args="--pretrained_egnn_feats_path $DATA_DIR/catpred_progres_embeds_dict.pt"
        fi

      echo "Starting prediction for parameter=$parameter and experiment=$dir..."
      python "$PREDICTION_SCRIPT" \
        --protein_records_path "$records_file" \
        --test_path "./$DATA_DIR/data/$parameter/${parameter}-random_test.csv" \
        --smiles_columns "$smiles_col" \
        --preds_path "$OUTPUT_DIR/$parameter/${parameter}_uncertainty_preds.csv" \
        --checkpoint_dir "$CKPT_DIR/${parameter}_ablation$retrain/${dir}" \
        --individual_ensemble_predictions --uncertainty_method "mve" \
        --batch_size 16 \
        $additional_args \
        # > "$LOG_DIR/${parameter}_uncertainty_preds.log" 2>&1
      echo "Prediction completed for parameter=$parameter and experiment=$dir. Logs saved to $LOG_DIR/${parameter}_uncertainty_preds.log"
    done
}

run_analysis() {
  for parameter in kcat km ki; do
    echo ">> Starting analysis for $parameter ..."
    python "$ANALYSIS_SCRIPT" \
        "$parameter" \
        "$OUTPUT_DIR/$parameter/${parameter}_uncertainty_preds.csv" \
        "$OUTPUT_DIR/${parameter}_uncertainty_analysis-summary.csv" \
        > "$LOG_DIR/${parameter}_uncertainty_analysis.log" 2>&1
    echo "   Analysis completed."
    echo "   Logs saved to $LOG_DIR/${parameter}_uncertainty_analysis.log"
  done
}

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
