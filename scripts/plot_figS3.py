import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = '../data/results/reproduce_results/'

def plot_all(metrics, outfile, retrained=False):
    # Define parameters, methods, and their respective colors
    params = ['ki']
    methods = ['seqemb36_attn6_ens10', 'seqemb36_attn6_esm_ens10', 
               'seqemb36_attn6_esm_ens10_Pretrained_egnnFeats', 
               'seqemb36_attn6_ens10_Pretrained_egnnFeats']
    exp_labels = {
        'seqemb36_attn6_ens10': 'Substrate + Seq-Attn',
        'seqemb36_attn6_esm_ens10': 'Substrate + Seq-Attn + pLM',
        'seqemb36_attn6_esm_ens10_Pretrained_egnnFeats': 'Substrate + Seq-Attn + pLM + EGNN', 
        'seqemb36_attn6_ens10_Pretrained_egnnFeats': 'Substrate + Seq-Attn + EGNN'
    }
    colors = ['#4593A5', '#003F5C', '#CAC2F6', '#000000']

    if retrained: 
        prefix = 'ablation_egnn_retrain'
    else: 
        prefix = 'ablation_egnn'
    
    # Define the categories and x-tick labels
    categories = ['Heldout', 'CLUSTER_99', 'CLUSTER_80', 'CLUSTER_60', 'CLUSTER_40']
    x_labels = ['100', '99', '80', '60', '40']
    
    # Initialize the figure and subplots (3 subplots for 3 metrics)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    # Loop over the metrics and plot them in separate subplots
    for j, metric in enumerate(metrics):
        ax = axes[j]  # Select subplot for the current metric
        ax.set_title(f'{metric} Comparison', fontsize=12, weight='bold')
        
        # Read and plot data for each method
        param_data = {}
        for method in methods:
            file_path = f'{RESULTS_DIR}/{params[0]}/{params[0]}_{method}_{prefix}_CatPredDB_CatPred_results.csv'
            df = pd.read_csv(file_path).set_index('Test')
            param_data[method] = df.loc[categories][[f'{metric}_mean', f'{metric}_stderr']]
        
        x = np.arange(len(categories))
        width = 0.2  # Width of each bar
        
        # Offset for each metric
        if metric == 'p1mag': offset = 1
        elif metric == 'R2': offset = 0.01
        else: offset = 0.03

        # Plot each method
        for i, method in enumerate(methods):
            metric_mean = param_data[method][f'{metric}_mean']
            metric_stderr = param_data[method][f'{metric}_stderr']
            
            for idx, (mean, stderr) in enumerate(zip(metric_mean, metric_stderr)):
                xpos = x[idx] + i * width
                
                if mean > 0:  # Only plot positive bars
                    bar = ax.bar(
                        xpos,
                        mean,
                        width,
                        color=colors[i],
                        yerr=stderr,
                        capsize=4
                    )
                    # Annotate the mean above the bar
                    ax.text(
                        xpos,
                        mean + stderr + offset,  # Small offset above error bar
                        f'{mean:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        rotation=90
                    )
                else:
                    # Place an asterisk at the bar's base for negative values
                    ax.text(
                        xpos,
                        0.01,  # Slightly above the x-axis
                        '*',
                        ha='center',
                        va='bottom',
                        fontsize=12,
                        color='black'
                    )
        
        # Set x-ticks and labels for each subplot
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(x_labels)
        
        # Set the y-label and title for the subplot
        ax.set_ylabel(f'{metric}', fontsize=12, weight='bold')
        ax.set_xlabel('Max. % seq. id. cutoff', fontsize=12, weight='bold')
        
        # Set the dynamic y-axis limit
        y_max = max([
            param_data[method][f'{metric}_mean'].max() + param_data[method][f'{metric}_stderr'].max()
            for method in methods
        ]) + 0.2

        if metric == 'p1mag': 
            y_max = 100
            
        ax.set_ylim(0, y_max)  # Fixed lower bound at 0, dynamic upper bound

    # Use exp_labels for legend (mapping methods to human-readable labels)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    labels = [exp_labels[label] for label in exp_labels]  # Use exp_labels for legend labels
    fig.legend(handles, labels, title='Method', loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4)
    
    # Adjust layout to avoid overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
    
    # Save and display the plot
    plt.savefig(outfile, dpi=300)
    print(f'Saved {outfile}')
    plt.show()

# Plot all metrics (R2, MAE, p1mag) in one figure
plot_all(['R2', 'MAE', 'p1mag'], f'../results/figS3.png')