import os
import sys
import pickle
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from sklearn.metrics import r2_score, mean_absolute_error
import ipdb
import csv

OUTPUT_DIR="../results/reproduce_results"
DATA_DIR = "../data/external/Baseline/"

def load_identity_data(parameter):
    """Load pre-calculated identity dictionary and mappings."""
    with open(f'{DATA_DIR}/{parameter}/{parameter}_test_train_identities_updated.pkl', 'rb') as f:
        data = pickle.load(f)
    train_seqs_dict = {val: key for key, val in data['train_seq_mapping'].items()}
    test_seqs_dict = {val: key for key, val in data['test_seq_mapping'].items()}
    return data, train_seqs_dict, test_seqs_dict

def load_train_data(data_path, parameter):
    """Load training data and define the label column based on the parameter."""
    label_col = f'log10{parameter}_mean' if parameter != 'kcat' else 'log10kcat_max'
    train_df = pd.read_csv(os.path.join(data_path, f'{parameter}-random_trainval.csv'))
    return train_df, label_col

def get_test_files(data_path, parameter, sim_cutoffs):
    """Generate the list of test files."""
    return [
        os.path.join(data_path, f'{parameter}-random_test.csv')
    ] + [
        os.path.join(data_path, f'{parameter}-random_test_sequence_{i}cluster.csv') for i in sim_cutoffs
    ], [ 'Cluster-100' ]+[ f'Cluster-{cutoff}' for cutoff in sim_cutoffs]

def get_top_pairs(identity_dict, train_seqs_dict, query_seq_id, n=3):
    """Get top N sequence pairs by identity for a query sequence ID."""
    pairs = [(train_seqs_dict[key[1]], value) for key, value in identity_dict.items() if key[0] == query_seq_id]
    pairs = list(set(pairs))  # Remove duplicates
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:n]

def compute_mean_labels(test_df, data, identity_dict, train_labels, train_seqs_dict, top_n=3):
    """Compute mean labels for test sequences based on top N similar sequences."""
    mean_labels = defaultdict(list)
    for test_seq in tqdm(test_df['sequence'], desc="Processing test sequences"):
        query_seq_id = data['test_seq_mapping'][test_seq]
        top_pairs = get_top_pairs(identity_dict, train_seqs_dict, query_seq_id, n=top_n)
        for target_seq, identity in top_pairs:
            label = train_labels.get(target_seq)
            if pd.notna(label):
                mean_labels[test_seq].append((identity, label, target_seq))
    return mean_labels

def update_test_df(test_df, mean_labels, label_col, top_n):
    """Update the test DataFrame with predicted labels and top N sequences."""
    for query, labels in mean_labels.items():
        if labels:
            mean_label = sum(label for _, label, _ in labels[:top_n]) / top_n
            test_df.loc[test_df['sequence'] == query, f'predicted_{label_col}'] = mean_label
            test_df.loc[test_df['sequence'] == query, f'top_{top_n}_seqs'] = ';'.join([seq for _, _, seq in labels[:top_n]])
            test_df.loc[test_df['sequence'] == query, f'top_{top_n}_seqids'] = ';'.join([str(round(identity, 3)) for identity, _, _ in labels[:top_n]])
        else:
            print(f"Warning: no labels found for query sequence '{query}'")
    return test_df

def compute_metrics(test_df, label_col):
    """Compute evaluation metrics for the test DataFrame."""
    test_df = test_df.dropna(subset=[f'predicted_{label_col}'])
    r2 = r2_score(test_df[label_col], test_df[f'predicted_{label_col}'])
    mae = mean_absolute_error(test_df[label_col], test_df[f'predicted_{label_col}'])
    absolute_errors = abs(test_df[label_col] - test_df[f'predicted_{label_col}'])
    p1mag = (absolute_errors < 1.0).mean() * 100
    return r2, mae, p1mag

def write_results_to_csv(results_dict, output_file):
    # Open the CSV file for writing
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the header row
        header = ['Test', 'R2_mean', 'R2_stderr', 'MAE_mean', 'MAE_stderr', 'p1mag_mean', 'p1mag_stderr']
        writer.writerow(header)
        
        # Write the data rows
        for i, (dataset, metrics) in enumerate(results_dict.items(), start=1):
            row = [
                dataset,
                metrics.get('R2', 0),         # Assign value to R2_mean
                0,                            # R2_stderr set to 0
                metrics.get('MAE', 0),        # Assign value to MAE_mean
                0,                            # MAE_stderr set to 0
                metrics.get('p1mag', 0),      # Assign value to p1mag_mean
                0                             # p1mag_stderr set to 0
            ]
            writer.writerow(row)
            
def main(PARAMETER, OUTPUT_FILE, recalculate=False):
    print(f"Processing parameter: {PARAMETER}")

    DATA_PATH = f'{DATA_DIR}/{PARAMETER}'
    SIM_CUTOFFS = [99, 80, 60, 40]
    TOP_N = 3

    # Load data
    data, train_seqs_dict, test_seqs_dict = load_identity_data(PARAMETER)
    train_df, label_col = load_train_data(DATA_PATH, PARAMETER)
    test_files, cluster_labels = get_test_files(DATA_PATH, PARAMETER, SIM_CUTOFFS)

    all_results = {}
    for i, test_file in enumerate(test_files):
        if i==0: 
            key = 'Heldout'
        else:
            key = f'CLUSTER_{SIM_CUTOFFS[i-1]}'
        if not recalculate:
            test_df = pd.read_csv(test_file[:-4]+'_mean_3sim.csv')
        else:
            # Process test files
            test_df = pd.read_csv(test_files[0])  # Assuming the main test file is first
            # Load pre-calculated identity dictionary
            data, train_seqs_dict, test_seqs_dict = load_identity_data(PARAMETER)
            
            identity_dict = data['data']
            mean_labels = compute_mean_labels(test_df, data, identity_dict, train_df[label_col].to_dict(), train_seqs_dict, top_n=TOP_N)
            test_df = update_test_df(test_df, mean_labels, label_col, top_n=TOP_N)
    
        # Compute metrics
        r2, mae, p1mag = compute_metrics(test_df, label_col)

        all_results[key] = {'R2':r2, 'MAE':mae, 'p1mag': p1mag}

    # Write results to file
    write_results_to_csv(all_results, OUTPUT_FILE)

if __name__ == "__main__":
    import sys
    PARAMETER = sys.argv[2].lower()
    OUTPUT_FILE = sys.argv[3]
    
    if sys.argv[1]=='recalculate':
        main(PARAMETER, OUTPUT_FILE, True)
    else:
        main(PARAMETER, OUTPUT_FILE, False)