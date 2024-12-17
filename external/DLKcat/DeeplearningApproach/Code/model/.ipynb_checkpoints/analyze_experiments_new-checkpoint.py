import sys
import os
import re
import numpy as np
from scipy import stats
import csv
import ipdb

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    epochs = []
    metrics = {
        'Test': {'MAE': [], 'R2': [], 'p1mag': []},
        'Dev40': {'MAE': [], 'R2': [], 'p1mag': []},
        'Dev60': {'MAE': [], 'R2': [], 'p1mag': []},
        'Dev80': {'MAE': [], 'R2': [], 'p1mag': []},
        'Dev99': {'MAE': [], 'R2': [], 'p1mag': []}
    }

    epoch_pattern = r'Epoch (\d+)'
    
    metric_pattern = r'{} MAE: ([\d.]+), {} R2: ([\d.-]+), {} p1mag: ([\d.]+)'

    for line in lines:
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            epochs.append(int(epoch_match.group(1)))

        for dataset in metrics.keys():
            match = re.search(metric_pattern.format(dataset, dataset, dataset), line)
            if match:
                metrics[dataset]['MAE'].append(float(match.group(1)))
                metrics[dataset]['R2'].append(float(match.group(2)))
                metrics[dataset]['p1mag'].append(float(match.group(3)))

    return epochs, metrics

def find_max_epoch(r2_values):
    if not r2_values:
        return 0  # Return 0 if no R2 values are available
    return r2_values.index(max(r2_values)) + 1  # Adding 1 because epochs are 1-indexed

def process_files(param, log_dir):
    results = []

    for seed in range(10):
        file_name = f"{log_dir}/{param}_{seed}_dim20.log"
        if os.path.exists(file_name):
            epochs, metrics = parse_log_file(file_name)
            if not epochs:  # Skip this file if no epochs were found
                print(f"Warning: No data found in {file_name}")
                continue
            
            optimal_epoch = find_max_epoch(metrics['Test']['R2'])
            result = {'epoch': optimal_epoch}
            
            for dataset, dataset_metrics in metrics.items():
                for metric, values in dataset_metrics.items():
                    if optimal_epoch <= len(values):
                        result[f'{dataset.lower()}_{metric.lower()}'] = values[optimal_epoch - 1]
                    else:
                        result[f'{dataset.lower()}_{metric.lower()}'] = values[-1]  # Use the last available value
                       
            results.append(result)

    return results


def calculate_stats(results):
    stats_results = {}

    print(len(results))
    for key in results[0].keys():
        values = [result[key] for result in results]
        mean = np.mean(values)
        sem = stats.sem(values)
        stats_results[key] = {'mean': mean, 'sem': sem}

    return stats_results

def write_results_to_tsv(param, stats_results, output_file):
    with open(output_file, 'w', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        
        # Write header
        writer.writerow(['c', 'R2', '', 'MAE', '', 'p1mag', ''])
        writer.writerow(['Metric', 'Mean', 'SEM', 'Mean', 'SEM', 'Mean', 'SEM'])
        
        # Define the order of datasets and map to the format in stats_results
        datasets = [
            ('Test', 'test'),
            ('CLUSTER_99', 'dev99'),
            ('CLUSTER_80', 'dev80'),
            ('CLUSTER_60', 'dev60'),
            ('CLUSTER_40', 'dev40')
        ]
        
        # Write data for each dataset
        for display_name, key_name in datasets:
            r2_mean = stats_results[f'{key_name}_r2']['mean']
            r2_sem = stats_results[f'{key_name}_r2']['sem']
            mae_mean = stats_results[f'{key_name}_mae']['mean']
            mae_sem = stats_results[f'{key_name}_mae']['sem']
            p1mag_mean = stats_results[f'{key_name}_p1mag']['mean']
            p1mag_sem = stats_results[f'{key_name}_p1mag']['sem']
            
            writer.writerow([
                display_name,
                f'{r2_mean:.4f}', f'{r2_sem:.4f}',
                f'{mae_mean:.4f}', f'{mae_sem:.4f}',
                f'{p1mag_mean:.4f}', f'{p1mag_sem:.4f}'
            ])

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python analyze_experiments_new.py <param> <logs_dir> <result_csv>")
        sys.exit(1)

    param = sys.argv[1]
    log_dir = sys.argv[2]
    output_file = sys.argv[3]
    
    results = process_files(param, log_dir)
    
    stats_results = calculate_stats(results)

    # Write results to TSV file
    write_results_to_tsv(param, stats_results, output_file)

    print(f"Results have been written to {output_file}")

    # Optionally, you can still print the results to console
    print(f"\nResults for parameter: {param}")
    for metric, values in stats_results.items():
        print(f"{metric.capitalize()}:")
        print(f"  Mean: {values['mean']:.4f}")
        print(f"  SEM: {values['sem']:.4f}")
