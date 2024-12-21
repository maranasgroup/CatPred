import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
import argparse
import sys
import os
import csv
import matplotlib.pyplot as plt

OUTPUT_DIR = '../results/'
# Define constants for cluster levels
CLUSTER_LEVELS = [99, 80, 60, 40]

def get_ood_indices(train_clusters, test_clusters):
    """
    Get indices of test samples that are out-of-distribution (OOD) based on clusters.

    Parameters:
        train_clusters (set): Set of cluster labels in the training data.
        test_clusters (pd.Series): Cluster labels in the test data.

    Returns:
        list: Indices of OOD samples.
    """
    return [i for i, cluster in enumerate(test_clusters) if cluster not in train_clusters]

def compute_metrics(y_true, y_pred):
    """
    Compute performance metrics for predictions.

    Parameters:
        y_true (pd.Series): Ground truth values.
        y_pred (pd.Series): Predicted values.

    Returns:
        dict: Dictionary containing MAE, R2, and p1mag metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    errors = np.abs(y_true - y_pred)
    p1mag = len(errors[errors < 1]) * 100 / len(errors)
    return {'MAE': mae, 'R2': r2, 'p1mag': p1mag}

def process_file(train_file, ground_truth_file, prediction_file, target_column_prefix):
    """
    Process a single prediction file to compute overall and OOD metrics.

    Parameters:
        train_file (str): Path to the training data file.
        ground_truth_file (str): Path to the ground truth data file.
        prediction_file (str): Path to the prediction file.
        target_column_prefix (str): Prefix of the target column.

    Returns:
        dict: Dictionary containing overall and OOD performance metrics.
    """
    # Load data
    train_df = pd.read_csv(train_file)
    gt_df = pd.read_csv(ground_truth_file)
    pred_df = pd.read_csv(prediction_file)

    # Merge ground truth and predictions
    smiles_column = 'reactant_smiles' if 'reactant_smiles' in gt_df.columns else 'substrate_smiles'
    merged_df = pd.merge(gt_df, pred_df, on=['sequence', smiles_column], suffixes=('_true', '_pred'))

    y_true = merged_df[f'{target_column_prefix}_true']
    y_pred = merged_df[f'{target_column_prefix}_pred']
    results = {'overall': compute_metrics(y_true, y_pred)}

    # Compute OOD performance for different cluster levels
    for N in CLUSTER_LEVELS:
        cluster_col = f'sequence_{N}cluster'
        merged_cluster_col = f'{cluster_col}_true'

        if cluster_col not in train_df.columns or merged_cluster_col not in merged_df.columns:
            print(f"Warning: {cluster_col} not found in the dataset.")
            continue

        train_clusters = set(train_df[cluster_col])
        test_clusters = merged_df[merged_cluster_col]
        ood_indices = get_ood_indices(train_clusters, test_clusters)

        if len(ood_indices) > 0:
            y_true_ood = y_true.iloc[ood_indices]
            y_pred_ood = y_pred.iloc[ood_indices]
            results[f'ood_cluster_{N}'] = compute_metrics(y_true_ood, y_pred_ood)
        else:
            print(f"Warning: No OOD samples found for cluster level {N}")

    return results

def calculate_stats(results_list):
    """
    Calculate mean and standard error of metrics across multiple runs.

    Parameters:
        results_list (list): List of results dictionaries from multiple runs.

    Returns:
        dict: Dictionary containing aggregated metrics.
    """
    stats_dict = {}
    for key in results_list[0].keys():
        stats_dict[key] = {}
        for metric in ['MAE', 'R2', 'p1mag']:
            values = [d[key][metric] for d in results_list]
            mean = np.mean(values)
            stderr = stats.sem(values)
            stats_dict[key][metric] = {'mean': mean, 'stderr': stderr}
    return stats_dict

def print_results(stats_dict):
    """
    Print summary of results in a readable format.

    Parameters:
        stats_dict (dict): Dictionary containing aggregated metrics.
    """
    print("Results Summary:")
    for dataset in stats_dict.keys():
        print(f"\n{dataset.upper()}:")
        print("-----------------")
        for metric in ['R2', 'MAE', 'p1mag']:
            mean = stats_dict[dataset][metric]['mean']
            stderr = stats_dict[dataset][metric]['stderr']
            print(f"{metric:<5}: {mean:.4f} ± {stderr:.4f}")

def output_results_csv(all_results, exp_names, output_file):
    """
    Save results to a CSV file.

    Parameters:
        all_results (list): List of dictionary objects containing aggregated metrics for each experiment
        exp_names (list): List of experiment names
        output_file (str): Path to the output CSV file
    """
    import csv

    # Function to write a metric to a CSV file
    def _write_metric_to_csv(metric_name, file_name):
        # Open the CSV file for writing
        with open(file_name, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write the header row
            header = ['Metric'] + exp_names  # First column is "Metric", rest are experiment names
            writer.writerow(header)
            
            # Determine all unique keys (like 'overall', 'ood_cluster_99', etc.)
            keys = list(all_results[0].keys())
            
            # Write rows for each key
            for key in keys:
                row = [key]  # First cell is the metric name (e.g., 'overall', 'ood_cluster_99', etc.)
                for result in all_results:
                    row.append(result[key][metric_name])  # Append the metric value
                writer.writerow(row)
        
    # Write CSV files for each metric
    _write_metric_to_csv('R2', output_file[:-4]+'_R2.csv')
    _write_metric_to_csv('MAE', output_file[:-4]+'_MAE.csv')
    _write_metric_to_csv('p1mag', output_file[:-4]+'_p1mag.csv')        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process prediction results for ML models.")
    parser.add_argument(
        "dataset_name", 
        choices=["kcat", "km", "ki"], 
        help="Dataset name determining the target column prefix."
    )
    parser.add_argument("train_file", help="Path to the training file.")
    parser.add_argument("ground_truth_file", help="Path to the ground truth test file.")
    parser.add_argument("output_file", help="Path to the output file for results.")
    parser.add_argument(
        "prediction_files", 
        nargs="+", 
        help="Paths to one or more prediction files."
    )
    
    args = parser.parse_args()
    
    # Determine target column prefix
    target_column_prefix = {
        "kcat": "log10kcat_max",
        "km": "log10km_mean",
        "ki": "log10ki_mean"
    }[args.dataset_name]

    # Process all prediction files
    all_results = [
        process_file(
            args.train_file, 
            args.ground_truth_file, 
            pred_file, 
            target_column_prefix
        ) for pred_file in args.prediction_files
    ]

    exp_names = []
    for file in args.prediction_files:
        exp_names.append(file.split('_exp')[-1][:-4])

    output_results_csv(all_results, exp_names, args.output_file)

    print(args.output_file)
    