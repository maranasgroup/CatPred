import os
import re
import numpy as np
import csv
import pandas as pd

# Function to extract individual model metrics and overall metrics from quiet.log file for each seed
def extract_metrics_for_seed(logfile, outdir, seed_name):
    model_maes = []
    model_r2s = []
    overall_mae = None
    overall_r2 = None

    # Extract Model MAE and R2 values for each model, and Overall MAE and R2
    with open(logfile, 'r') as file:
        for line in file:
            model_mae_match = re.search(r'Model \d+ test mae = ([0-9.]+)', line)
            model_r2_match = re.search(r'Model \d+ test r2 = ([0-9.]+)', line)
            overall_mae_match = re.search(r'Overall test mae = ([0-9.]+)', line)
            overall_r2_match = re.search(r'Overall test r2 = ([0-9.]+)', line)

            if model_mae_match:
                model_maes.append(float(model_mae_match.group(1)))
            if model_r2_match:
                model_r2s.append(float(model_r2_match.group(1)))
            if overall_mae_match:
                overall_mae = float(overall_mae_match.group(1))
            if overall_r2_match:
                overall_r2 = float(overall_r2_match.group(1))

    # Calculate Mean and SEM for Model MAE and Model R2
    model_mae_mean = np.mean(model_maes)
    model_mae_sem = np.std(model_maes, ddof=1) / np.sqrt(len(model_maes))
    
    model_r2_mean = np.mean(model_r2s)
    model_r2_sem = np.std(model_r2s, ddof=1) / np.sqrt(len(model_r2s))
    
    # Append results to the output CSV
    with open(os.path.join(outdir, 'ensemble_analysis_summary.csv'), mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([seed_name, model_r2_mean, model_r2_sem, overall_r2, model_mae_mean, model_mae_sem, overall_mae])

    # Return DataFrame to aggregate results later
    return pd.read_csv(os.path.join(outdir, 'ensemble_analysis_summary.csv'))

# Process each directory (kcat, km, ki) and extract metrics for each seed
def process_metric_directory(metric_dir, subdir_suffix):
    # Output DataFrame to accumulate results
    all_seeds_df = pd.DataFrame()

    # Loop through each seed directory for the current metric
    for seed_dir in os.listdir(metric_dir):
        if seed_dir.startswith('seed'):
            seed_name = seed_dir
            log_file = os.path.join(metric_dir, seed_dir, subdir_suffix, 'quiet.log')
            if os.path.isfile(log_file):
                df = extract_metrics_for_seed(log_file, metric_dir, seed_name)
                all_seeds_df = pd.concat([all_seeds_df, df], ignore_index=True)

    return all_seeds_df

# Function to calculate mean and std for individual and ensemble metrics
def calculate_and_save_summary(kcat_df, km_df, ki_df, output_file):
    # Calculate Mean and Stdev for Individual and Ensemble metrics
    def calc_metrics(df):
        individual_r2 = df['Mean_R2']
        individual_mae = df['Mean_MAE']
        ensemble_r2 = df['Overall_R2']
        ensemble_mae = df['Overall_MAE']
        
        individual_r2_mean = np.mean(individual_r2)
        individual_r2_std = np.std(individual_r2)
        ensemble_r2_mean = np.mean(ensemble_r2)
        ensemble_r2_std = np.std(ensemble_r2)
        
        individual_mae_mean = np.mean(individual_mae)
        individual_mae_std = np.std(individual_mae)
        ensemble_mae_mean = np.mean(ensemble_mae)
        ensemble_mae_std = np.std(ensemble_mae)
        
        return {
            "R2 Mean": [individual_r2_mean, ensemble_r2_mean],
            "R2 Stdev.": [individual_r2_std, ensemble_r2_std],
            "MAE Mean": [individual_mae_mean, ensemble_mae_mean],
            "MAE Stdev.": [individual_mae_std, ensemble_mae_std]
        }

    # Calculate metrics for each dataset
    kcat_metrics = calc_metrics(kcat_df)
    km_metrics = calc_metrics(km_df)
    ki_metrics = calc_metrics(ki_df)

    # Print the final table
    print(f"{'Metric':<20}{'Individual':<15}{'Ensemble':<15}{'Individual':<15}{'Ensemble':<15}{'Individual':<15}{'Ensemble':<15}")
    
    # Printing R2 Mean, R2 Stdev., MAE Mean, and MAE Stdev.
    for metric, values in kcat_metrics.items():
        print(f"{metric:<20}{values[0]:<15.4f}{values[1]:<15.4f}{km_metrics[metric][0]:<15.4f}{km_metrics[metric][1]:<15.4f}{ki_metrics[metric][0]:<15.4f}{ki_metrics[metric][1]:<15.4f}")

    
    # Prepare the results for CSV
    table_data = []

    # Add rows for R2 Mean, R2 Stdev., MAE Mean, and MAE Stdev.
    for metric, values in kcat_metrics.items():
        row = [metric,
               values[0], values[1],
               km_metrics[metric][0], km_metrics[metric][1],
               ki_metrics[metric][0], ki_metrics[metric][1]]
        table_data.append(row)

    # Write the table to CSV
    header = ['Metric', 'CatPred-kcat Individual', 'CatPred-kcat Ensemble', 
              'CatPred-Km Individual', 'CatPred-Km Ensemble', 
              'CatPred-Ki Individual', 'CatPred-Ki Ensemble']
    
    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(table_data)

    print(f"Summary saved to {output_file}")

# Main processing for each metric directory
def main():
    kcat_df = process_metric_directory("../data/pretrained/reproduce_checkpoints/kcat", "seqemb36_attn6_esm_ens10")
    km_df = process_metric_directory("../data/pretrained/reproduce_checkpoints/km", "seqemb36_attn6_esm_ens10")
    ki_df = process_metric_directory("../data/pretrained/reproduce_checkpoints/ki", "seqemb36_attn6_ens10_Pretrained_egnnFeats")
    
    output_file = "../results/TableS10.csv"
    calculate_and_save_summary(kcat_df, km_df, ki_df, output_file)

if __name__ == "__main__":
    main()
