#!/bin/bash

# Function to extract individual model metrics and overall metrics from quiet.log file for each seed
extract_metrics_for_seed() {
    local logfile=$1
    local outdir=$2
    local seed_name=$3
    local model_maes=()
    local model_r2s=()
    local overall_mae overall_r2

    # Extract Model MAE and R2 values for each model, and Overall MAE and R2
    while IFS= read -r line; do
        if [[ $line =~ Model\ [0-9]+\ test\ mae\ =\ ([0-9.]+) ]]; then
            model_maes+=("${BASH_REMATCH[1]}")
        elif [[ $line =~ Model\ [0-9]+\ test\ r2\ =\ ([0-9.]+) ]]; then
            model_r2s+=("${BASH_REMATCH[1]}")
        elif [[ $line =~ Overall\ test\ mae\ =\ ([0-9.]+) ]]; then
            overall_mae="${BASH_REMATCH[1]}"
        elif [[ $line =~ Overall\ test\ r2\ =\ ([0-9.]+) ]]; then
            overall_r2="${BASH_REMATCH[1]}"
        fi
    done < "$logfile"

    # Calculate Mean and SEM for Model MAE and Model R2
    model_mae_mean=$(printf "%s\n" "${model_maes[@]}" | awk '{sum+=$1; sumsq+=$1*$1} END {print sum/NR}')
    model_mae_sem=$(printf "%s\n" "${model_maes[@]}" | awk -v mean="$model_mae_mean" '{sumsq+=($1-mean)^2} END {print sqrt(sumsq/(NR*(NR-1)))}')
    
    model_r2_mean=$(printf "%s\n" "${model_r2s[@]}" | awk '{sum+=$1; sumsq+=$1*$1} END {print sum/NR}')
    model_r2_sem=$(printf "%s\n" "${model_r2s[@]}" | awk -v mean="$model_r2_mean" '{sumsq+=($1-mean)^2} END {print sqrt(sumsq/(NR*(NR-1)))}')
    
    # Append results to the output CSV
    echo "$seed_name,$model_r2_mean,$model_r2_sem,$overall_r2,$model_mae_mean,$model_mae_sem,$overall_mae" >> "$outdir/summary.csv"
}

# Process each directory (kcat, km, ki) and extract metrics for each seed
process_metric_directory() {
    local metric_dir=$1
    local subdir_suffix=$2
    local output_file="$metric_dir/summary.csv"

    # Prepare CSV header
    echo "Seed,Mean_R2,SEM_R2,Overall_R2,Mean_MAE,SEM_MAE,Overall_MAE" > "$output_file"

    # Loop through each seed directory for the current metric
    for seed_dir in "$metric_dir/checkpoints"/*; do
        seed_name=$(basename "$seed_dir")
        log_file="$seed_dir/$subdir_suffix/quiet.log"
        if [[ -f $log_file ]]; then
            extract_metrics_for_seed "$log_file" "$metric_dir" "$seed_name"
        fi
    done
}

# Process each directory with the correct sub-directory path for each metric
process_metric_directory "kcat" "seqemb36_attn6_esm_ens10"
process_metric_directory "km" "seqemb36_attn6_esm_ens10"
process_metric_directory "ki" "seqemb36_attn6_ens10"

echo "CSV files created for kcat, km, and ki with comparisons of model and overall metrics in each."
