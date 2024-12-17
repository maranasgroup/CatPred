import csv
import re

def parse_log_and_save_csv(log_file, output_file):
    # Pattern to match dataset headers and metrics
    dataset_pattern = re.compile(r"^(Heldout Dataset|CLUSTER_\d+):?")
    metric_pattern = re.compile(r"^(\w+)\s*:\s*([\d.]+)\s*±\s*([\d.]+)")
    
    # To store parsed results
    results = []
    current_dataset = None

    # Read the log file
    with open(log_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            # Check for dataset header
            dataset_match = dataset_pattern.match(line)
            if dataset_match:
                # Handle "Heldout Dataset" header
                if dataset_match.group(1) == "Heldout Dataset":
                    current_dataset = "Heldout"
                else:
                    current_dataset = dataset_match.group(1)
                continue
            
            # Check for metric values (R2, MAE, p1mag)
            metric_match = metric_pattern.match(line)
            if metric_match and current_dataset:
                metric_name = metric_match.group(1)
                mean_value = float(metric_match.group(2))
                stderr_value = float(metric_match.group(3))
                if metric_name=='p1mag': 
                    mean_value = mean_value *100
                    stderr_value = stderr_value *100
                
                # Add or update the results for the current dataset
                if not any(res['name'] == current_dataset for res in results):
                    results.append({'name': current_dataset})
                for res in results:
                    if res['name'] == current_dataset:
                        res[f"{metric_name}_mean"] = mean_value
                        res[f"{metric_name}_stderr"] = stderr_value

    # Write to CSV
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Header row
        header = ['Test'] + ['R2_mean', 'R2_stderr', 'MAE_mean', 'MAE_stderr', 'p1mag_mean', 'p1mag_stderr']
        writer.writerow(header)
        
        # Data rows
        for i, result in enumerate(results, start=1):
            row = [
                result['name'],
                result.get('R2_mean', ''),
                result.get('R2_stderr', ''),
                result.get('MAE_mean', ''),
                result.get('MAE_stderr', ''),
                result.get('p1mag_mean', ''),
                result.get('p1mag_stderr', ''),
            ]
            writer.writerow(row)

import sys

log_file = sys.argv[1]
output_file = sys.argv[2]
parse_log_and_save_csv(log_file, output_file)
