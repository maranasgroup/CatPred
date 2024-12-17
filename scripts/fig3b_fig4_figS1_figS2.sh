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
ANALYSIS_SCRIPT="./scripts/analyze_ablation.py"

# Ensure directories exist
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$OUTPUT_DIR"

# Define functions
run_training() {
  directories=(
        substrate_only
        seqemb36_attn6_ens10
        seqemb36_attn6_esm_ens10
        seqemb36_attn6_esm_ens10_Pretrained_egnnFeats
    )
  for parameter in ki kcat km; do
      smiles_col="reactant_smiles"
      records_file="./$DATA_DIR/data/$parameter/${parameter}_max_wt_singleSeqs_wpdbs_pdbrecords.json.gz"
        
      if [ "$parameter" != "kcat" ]; then
        smiles_col="substrate_smiles"
        records_file="./$DATA_DIR/data/$parameter/${parameter}_mean_wt_singleSeqs_wpdbs_pdbrecords.json.gz"
      fi
    
    # Loop over the directories
    for dir in "${directories[@]}"; do
        echo "Processing experiment: $dir"

        if [ "$dir" = "substrate_only" ]; then
            additional_args="--skip_protein"
        fi
        if [ "$dir" = "seqemb36_attn6_ens10" ]; then
            additional_args="--ensemble_size 10"
        fi
        if [ "$dir" = "substrate_only" ]; then
            additional_args="--add_esm_feats"
        fi
        if [ "$dir" = "substrate_only" ]; then
            additional_args="--add_esm_feats --add_pretrained_egnn_feats --pretrained_egnn_feats_path $DATA_DIR/catpred_progres_embeds_dict.pt"
        fi
      
      echo "Starting training for parameter=$parameter and exp=$dir..."
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
        --save_dir "$CKPT_DIR/${parameter}_ablation_retrain/${exp}" --epochs 30 \
        $additional_args \
        # > "$LOG_DIR/${parameter}_training_trainvalModel_seed${seed}.log" 2>&1
    done
  done
}

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
      smiles_col="reactant_smiles"
      records_file="./$DATA_DIR/data/$parameter/${parameter}_max_wt_singleSeqs_wpdbs_pdbrecords.json.gz"
        
      if [ "$parameter" != "kcat" ]; then
        smiles_col="substrate_smiles"
        records_file="./$DATA_DIR/data/$parameter/${parameter}_mean_wt_singleSeqs_wpdbs_pdbrecords.json.gz"
      fi
      
        # Loop over the directories
        for dir in "${directories[@]}"; do
            echo "Processing experiment: $dir"

            if [ "$dir" = "seqemb36_attn6_esm_ens10_Pretrained_egnnFeats" ]; then
                additional_args="--pretrained_egnn_feats_path $DATA_DIR/catpred_progres_embeds_dict.pt"
            fi
          
          echo "Starting prediction for parameter=$parameter and experiment=$dir..."
          python "$PREDICTION_SCRIPT" \
            --protein_records_path "$records_file" \
            --test_path "./$DATA_DIR/data/$parameter/${parameter}-random_test.csv" \
            --smiles_columns "$smiles_col" \
            --preds_path "$OUTPUT_DIR/$parameter/${parameter}_ablation_trainvalModel_exp${dir}.csv" \
            --checkpoint_dir "$CKPT_DIR/${parameter}_ablation$retrain/${dir}" \
            --individual_ensemble_predictions \
            --batch_size 32 \
            $additional_args \
            > "$LOG_DIR/${parameter}_ablation_trainvalModel_exp${dir}.log" 2>&1
          echo "Prediction completed for parameter=$parameter and experiment=$dir. Logs saved to $LOG_DIR/${parameter}_ablation_trainvalModel_exp${dir}.log"
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
        "$OUTPUT_DIR/${parameter}_ablation_analysis-summary.csv" \
        $OUTPUT_DIR/$parameter/${parameter}_ablation_trainvalModel* \
        # > "$LOG_DIR/${parameter}_ablation_analysis_trainvalModel.log" 2>&1
    echo "   Analysis completed. Results saved to $OUTPUT_DIR/${parameter}_ablation_analysis-summary_R2.csv, $OUTPUT_DIR/${parameter}_ablation_analysis-summary_MAE.csv and $OUTPUT_DIR/${parameter}_ablation_analysis-summary_p1mag.csv"
    echo "   Logs saved to $LOG_DIR/${parameter}_ablation_analysis_trainvalModel.log"
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
    run_prediction _retrain
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
