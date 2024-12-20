import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = '../data/results/reproduce_results/'

def plot_all(metric, outfile):
    # Define parameters, methods, and their respective colors
    params = ['kcat', 'km', 'ki']
    methods = ['Baseline', 'DLKcat', 'UniKP', 'CatPred']
    colors = ['#F0E442', '#F2AA84', '#009E73', '#0072B2']
    
    # Define the categories and x-tick labels
    categories = ['Heldout', 'CLUSTER_99', 'CLUSTER_80', 'CLUSTER_60', 'CLUSTER_40']
    x_labels = ['100', '99', '80', '60', '40']
    
    # Initialize the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # Read and plot data for each parameter
    for ax, param in zip(axes, params):
        # Read data for all methods
        param_data = {}
        for method in methods:
            file_path = f'{RESULTS_DIR}/{param}/{param}_CatPredDB_{method}_results.csv'
            df = pd.read_csv(file_path).set_index('Test')
            param_data[method] = df.loc[categories][[f'{metric}_mean', f'{metric}_stderr']]
        
        # Plot bars for each method
        x = np.arange(len(categories))
        width = 0.2  # Width of each bar

        if metric=='p1mag': offset = 1
        elif metric=='R2': offset = 0.01
        else: offset = 0.03
        for i, method in enumerate(methods):
            r2_mean = param_data[method][f'{metric}_mean']
            r2_stderr = param_data[method][f'{metric}_stderr']
            
            for idx, (mean, stderr) in enumerate(zip(r2_mean, r2_stderr)):
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
        
        # Set subplot titles and x-ticks
        ax.set_title(f'CatPred-DB-{param}', fontsize=12, weight='bold')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(x_labels)

    if metric=='R2': lat = r'R$^2$'
    elif metric=='MAE': lat = 'MAE'
    else: lat = r'$p_{1mag}$'
    # Formatting
    fig.supylabel(lat, fontsize=12, weight='bold')  # Common y-axis title with superscript 2
    fig.supxlabel('Max. % seq. id. cutoff', fontsize=12, weight='bold')  # Common x-axis title
    
    # Add a legend outside the first subplot
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
    labels = methods
    fig.legend(handles, labels, title='Method', loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
    # Dynamically calculate the upper bound
    y_max = max([
        param_data[method][f'{metric}_mean'].max() + param_data[method][f'{metric}_stderr'].max()
        for method in methods
    ]) + 0.2

    if metric=='p1mag': y_max = 100
        
    plt.ylim(0, y_max)  # Fixed lower bound at 0, dynamic upper bound

    plt.savefig(outfile, dpi=300)
    print('Saved', outfile)
    plt.show()

plot_all('R2', f'../results/fig5.png')
plot_all('MAE', f'../results/figS7_abc.png')
plot_all('p1mag', f'../results/figS7_def.png')
