import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, mean_squared_error
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import glob

# Error calculation function
def _error_calc(target, pred):
    """
    Calculate the absolute errors and cumulative percentage of errors.

    Parameters:
    target (list): List of true target values.
    pred (list): List of predicted values.

    Returns:
    tuple: Cumulative percentage of errors at an error of 1, bin edges, and cumulative percentages.
    """
    errors = np.abs(np.array(target) - np.array(pred))
    bins = np.arange(0, max(errors) + 0.1, 0.1)
    freqs, bin_edges = np.histogram(errors, bins)
    percs = 100 * freqs / len(errors)
    cum_percs = np.cumsum(percs)

    try:
        index_err1 = np.where(bin_edges == 1.0)[0][0]
        cum_perc_err1 = cum_percs[index_err1]
    except IndexError:
        cum_perc_err1 = None

    return cum_perc_err1, bin_edges[1:], cum_percs

# Bin by standard deviation
def _bin_by_std(target, pred, std, cutoff):
    """
    Bin data by standard deviation.

    Parameters:
    target (list): List of true target values.
    pred (list): List of predicted values.
    std (list): List of standard deviation values.
    cutoff (float): Cutoff value for standard deviation.

    Returns:
    DataFrame: Filtered DataFrame containing target, predicted, and standard deviation columns.
    """
    df = pd.DataFrame({'target': target, 'pred': pred, 'stdev': std})
    return df[df.stdev <= cutoff]

# Calculate evaluation metrics
def _calc_metrics(target, pred, std):
    """
    Calculate various metrics for model evaluation.

    Parameters:
    target (list): List of true target values.
    pred (list): List of predicted values.
    std (list): List of standard deviation values.
    R (int): Range or cutoff value.

    Returns:
    tuple: Metrics and metrics categorized by standard deviation.
    """
    quartiles = np.percentile(std, [25, 50, 75, 100])
    std_bins = [float(q) for q in quartiles]
    cum_perc_err1 = {}
    metrics_std = {'r2': {}, 'mae': {}, 'p1mag': {}}
    target_linear = np.power(10, target)
    pred_linear = np.power(10, pred)
    
    for bin in std_bins:
        df = _bin_by_std(target, pred, std, bin)
        target_, pred_ = df.target, df.pred
        errors_ = np.abs(target_ - pred_)
        
        y_mean = np.mean(target_)

        # Calculate SS_res (Residual Sum of Squares)
        SS_res = np.sum((target_ - pred_) ** 2)
        # Calculate SS_tot (Total Sum of Squares)
        SS_tot = np.sum((target_ - y_mean) ** 2)

        cum_perc_err1[bin], bins, cums = _error_calc(target_, pred_)
        metrics_std['r2'][bin] = r2_score(target_, pred_)
        metrics_std['p1mag'][bin] = len(errors_[errors_ < 1]) *100 / len(errors_)
        metrics_std['mae'][bin] = mean_absolute_error(target_, pred_)

    
    return metrics_std

def get_ood_indices(train_clusters, test_clusters):
    return [i for i, cluster in enumerate(test_clusters) if cluster not in train_clusters]

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    errors = np.abs(y_true - y_pred)
    p1mag = len(errors[errors < 1]) *100 / len(errors)
    r = stats.pearsonr(y_true, y_pred)[0]
    rmse = root_mean_squared_error(y_true, y_pred)

    return {'MAE': mae, 'R2': r2, 'p1mag': p1mag}

def process_file(train_file, ground_truth_file, prediction_file, parameter):
    # Load train, ground truth and predictions
    train_df = pd.read_csv(train_file)
    gt_df = pd.read_csv(ground_truth_file)
    pred_df = pd.read_csv(prediction_file)

    smicol = 'reactant_smiles' if parameter=='kcat' else 'substrate_smiles'
    labelcol = f'log10{parameter}_max' if parameter=='kcat' else f'log10{parameter}_mean'
    stdevcol = f'{labelcol}_mve_uncal_var'
    
    # Merge ground truth and predictions based on 'sequence' and 'reactant_smiles'
    merged_df = pd.merge(gt_df, pred_df, on=['sequence', smicol], suffixes=('_true', '_pred'))
    
    print(len(gt_df), len(pred_df), len(merged_df)-len(pred_df))
    y_true = merged_df[f'{labelcol}_true']
    y_pred = merged_df[f'{labelcol}_pred']
    std = merged_df[stdevcol]
    metrics_std100 = _calc_metrics(y_true, y_pred, std)

    # print('-' * 50)
    # print('Heldout')
    # print('-' * 50)
    # for metric in metrics_std100:
    #     print(f'{metric}:')
    #     for bin in metrics_std100[metric]:
    #         print(f'{bin}, {metrics_std100[metric][bin]}')
        
    # Compute OOD performance for different cluster levels
    for N in [99]:
        cluster_col = f'sequence_{N}cluster'
        merged_cluster_col = f'{cluster_col}_true'  # The merged column name
        
        if not cluster_col in train_df.columns or not merged_cluster_col in merged_df.columns:
            print(f"Warning: {cluster_col} not found in the dataset.")
            continue
        
        train_clusters = set(train_df[cluster_col])
        test_clusters = merged_df[merged_cluster_col]
        
        ood_indices = get_ood_indices(train_clusters, test_clusters)
        
        y_true_ood = y_true.iloc[ood_indices]
        y_pred_ood = y_pred.iloc[ood_indices]
        std_ood = std.iloc[ood_indices]
        
        metrics_std99 = _calc_metrics(y_true_ood, y_pred_ood, std_ood)
        # print('-' * 50)
        # print('OOD 99')
        # print('-' * 50)
        # for metric in metrics_std99:
        #     print(f'{metric}:')
        #     for bin in metrics_std99[metric]:
        #         print(f'{bin}, {metrics_std99[metric][bin]}')
        
    return metrics_std100, metrics_std99

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Fixed colors for bins
BIN_COLORS = ["#003F5C", "#568F8B", "#B9D5B2", "#F4F7CC"]

# Helper function to save metrics to CSV
def save_metrics_to_csv(metrics_std100, metrics_std99, parameter, output_dir):
    """
    Save metrics_std100 and metrics_std99 to CSV files.
    """
    # Convert to DataFrame and save
    for metrics, name in zip([metrics_std100, metrics_std99], ['std100', 'std99']):
        if name=='std100': savename = 'Heldout'
        else: savename = 'ood_cluster99'    
        output_file = f"{output_dir}/reproduce_results/{parameter}/{parameter}_uncertainty_analysis_{savename}.csv"
        rows = []
        for metric, bins in metrics.items():
            for bin_val, value in bins.items():
                rows.append({'metric': metric, 'SD percentile': bin_val, 'value': value})
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

# Helper function to prepare DataFrame for plotting
def prepare_plot_data(metrics_dict, metric_name):
    """
    Prepare DataFrame for plotting from metrics dictionary.
    """
    plot_data = []
    for param, metrics in metrics_dict.items():
        if metric_name in metrics:
            for std_bin, value in metrics[metric_name].items():
                plot_data.append({'Parameter': param, 'SD percentile': std_bin, 'Value': value})
                
    df = pd.DataFrame(plot_data)

    # Step 1: Apply bin mapping separately for each parameter
    def map_bins_for_param(param_df):
        # Get unique sorted bin values dynamically for each parameter
        unique_bins = sorted(param_df['SD percentile'].unique(), reverse=True)
        
        # Map sorted float bins to bin names
        bin_names = ["100", "75", "50", "25"]
        bin_mapping = {bin_val: bin_name for bin_val, bin_name in zip(unique_bins, bin_names)}
        
        # Replace float bins with bin names for this specific parameter
        param_df['SD percentile'] = param_df['SD percentile'].map(bin_mapping)
        
        # Convert to ordered categorical with consistent bin ordering
        param_df['SD percentile'] = pd.Categorical(param_df['SD percentile'], ordered=True, categories=bin_names)
        
        return param_df

    # Apply bin mapping for each unique parameter
    df = df.groupby('Parameter').apply(map_bins_for_param)

    return df

# Function to plot merged metrics in a single figure
def plot_merged_metrics(metrics_std100, metrics_std99, output_dir):
    """
    Plot merged R2, MAE, and p1mag metrics for metrics_std100 and metrics_std99 side by side.
    """
    metrics = ['r2', 'mae', 'p1mag']  # Metrics to plot
    parameters = ['kcat', 'km', 'ki']  # Parameters
    titles = ['Metrics_std100', 'Metrics_std99']
    
    # Create two figures: one for std100 and one for std99
    for i, metrics_data in enumerate([metrics_std100, metrics_std99]):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        # fig.suptitle(f"{titles[i]}: R², MAE, and p1mag Metrics", fontsize=16)

        for ax, metric in zip(axes, metrics):
            # Prepare data
            df = prepare_plot_data(metrics_data, metric)
            
            # Plot barplot
            sns.barplot(
                data=df, x='Parameter', y='Value', hue='SD percentile', palette=BIN_COLORS, ax=ax
            )

            ax.legend_.remove()
            
            # ax.set_title(metric.upper())
            ax.set_xlabel("Parameter")
            if metric=='r2': 
                lat = r'R$^2$'
                ax.set_ylim(0, 1)
            elif metric=='mae': 
                lat = 'MAE'
                ax.set_ylim(0, 1.5)
            elif metric=='p1mag': 
                lat = r'$p_{1mag}$'
                ax.set_ylim(0, 100)
            ax.set_ylabel(lat)
            # Add annotations
            for p in ax.patches:
                height = p.get_height()
                if height > 1e-3:
                    ax.annotate(
                        f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        xytext=(0, 8), 
                        textcoords='offset points', 
                        ha='center', 
                        va='bottom', 
                        rotation=90  # Anti-clockwise rotation by 90 degrees
                    )

        # Add a common legend for all subplots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, title="SD percentile")

        print(titles[i])
        # Save figure
        if 'std99' in titles[i]: 
            output_file = f"{output_dir}/figS8.png"
        if 'std100' in titles[i]: 
            output_file = f"{output_dir}/fig6.png"
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_file)
        print(f"Saved plot: {output_file}")
        plt.close()

# Main function to process and plot all metrics
def main():
    parameters = ['kcat', 'km', 'ki']
    output_dir = "../results/"  # Output directory for plots
    
    all_metrics_std100 = {}
    all_metrics_std99 = {}
    
    # Loop through each parameter and process files
    for parameter in parameters:
        prediction_file = f"../results/reproduce_results/{parameter}/{parameter}_uncertainty_preds.csv" 
        train_file = f'../data/CatPred-DB/data/{parameter}/{parameter}-random_trainval.csv'
        ground_truth_file = f'../data/CatPred-DB/data/{parameter}/{parameter}-random_test.csv'
        
        # Process metrics
        metrics_std100, metrics_std99 = process_file(train_file, ground_truth_file, prediction_file, parameter)

        # Save metrics to CSV
        save_metrics_to_csv(metrics_std100, metrics_std99, parameter, output_dir)
        
        # Store metrics for later visualization
        all_metrics_std100[parameter] = metrics_std100
        all_metrics_std99[parameter] = metrics_std99
    
    # Plot merged metrics
    plot_merged_metrics(all_metrics_std100, all_metrics_std99, output_dir)

if __name__ == '__main__':
    main()
