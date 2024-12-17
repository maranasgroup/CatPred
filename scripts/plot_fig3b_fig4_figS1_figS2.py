import matplotlib.pyplot as plt
import csv
import os
import numpy as np

# Plot Horizontal Bar Chart for multiple parameters
def plot_r2_results(file_name, metric, ax):
    exp_labels = {
        'Substrate Only': 'substrate_only',
        '+ Seq-Attn': 'seqemb36_attn6_ens10',
        '+ pLM': 'seqemb36_attn6_esm_ens10',
        '+ EGNN': 'seqemb36_attn6_esm_ens10_Pretrained_egnnFeats'
    }
        
    # Read the R2 data
    data = {}
    with open(file_name, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            metric = row[0]
            data[metric] = {h: float(val) for val, h in zip(row[1:], header[1:])}
    
    # Set up categories
    categories = ['+ EGNN', '+ pLM', '+ Seq-Attn', 'Substrate Only']

    # Extract overall and cluster_99 R2 values
    overall = [data['overall'][exp_labels[h]] for h in categories]
    cluster_99 = [data['ood_cluster_99'][exp_labels[h]] for h in categories]
    
    y_pos = np.arange(len(categories))
    bar_height = 0.3  # Reduce bar height for spacing
    colors = ['#CAC2F6', '#003F5C', '#4493A5', '#E67C69']  # Colors
    hatch_style = ['\\\\', '\\\\', '\\\\', '\\\\']  # For dashed filling

    # Plot overall R2 (solid bars)
    for i, (val, color) in enumerate(zip(overall, colors)):
        ax.barh(y_pos[i] + bar_height/2, val, color=color, edgecolor='white', height=bar_height, label=None)
        ax.text(val + 0.01, y_pos[i] + bar_height/2, f'{val:.3f}', va='center')

    # Plot cluster_99 R2 (bars with hatching)
    for i, (val, color, hatch) in enumerate(zip(cluster_99, colors, hatch_style)):
        ax.barh(y_pos[i] - bar_height/2, val, color=color, edgecolor='white', hatch=hatch, height=bar_height, label=None)
        ax.text(val + 0.01, y_pos[i] - bar_height/2, f'{val:.3f}', va='center')

    # Set axis and labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlim(0, max(max(overall), max(cluster_99)) + 0.1)
    if metric == 'R2':
        ax.set_xlabel('R²')
    elif metric == 'MAE':
        ax.set_xlabel('MAE')
    elif metric == 'p1mag':
        ax.set_xlabel(r'$p_1^{mag}$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set title for the subplot
    param_name = os.path.basename(file_name).split('_')[0]  # kcat, km, ki
    ax.set_title(f'CatPred-{param_name}', fontsize=12, weight='bold')

# Set up figure for three adjacent plots
def plot_all(metric, outfile):
    OUTDIR = '../results/reproduce_results/'
    params = ['kcat', 'km', 'ki']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Three plots side by side
    
    # Plot data for each parameter on a different axis
    for ax, param in zip(axes, params):
        file_name = f'{OUTDIR}/{param}_ablation_analysis-summary_{metric}.csv'
        plot_r2_results(file_name, metric, ax)
    
    # Add a legend outside the plots
    handles = [plt.Rectangle((0,0),1,1, color=color) for color in ['#CAC2F6', '#003F5C', '#4493A5', '#E67C69']]
    labels = ['+ EGNN', '+ pLM', '+ Seq-Attn', 'Substrate Only']
    fig.legend(handles, labels, title='Method', loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
    plt.savefig(outfile, dpi=300)

outfile = f'../results/fig3b.png'
metric = 'R2'
# Run the plotting function
plot_all(metric, outfile)

outfile = f'../results/figS1a.png'
metric = 'MAE'
# Run the plotting function
plot_all(metric, outfile)

outfile = f'../results/figS1b.png'
metric = 'p1mag'
# Run the plotting function
plot_all(metric, outfile)

# Plot Vertical Bar Chart with Adjacent Bars and Value Labels
def plot_r2_vertical_results(file_name, metric):
    exp_labels = {
        'Substrate Only': 'substrate_only',
        '+ Seq-Attn': 'seqemb36_attn6_ens10',
        '+ pLM': 'seqemb36_attn6_esm_ens10',
        '+ EGNN': 'seqemb36_attn6_esm_ens10_Pretrained_egnnFeats'
    }
        
    # Read the R2 data
    data = {}
    with open(file_name, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            metric = row[0]
            data[metric] = {h: float(val) for val, h in zip(row[1:], header[1:])}
    
    # Set up figure
    categories = ['cluster_99', 'cluster_80', 'cluster_60', 'cluster_40']
    categories_labels = ['99', '80', '60', '40']

    # Extract R2 values for each category and experiment
    cluster_99 = [data['ood_cluster_99'][exp_labels[h]] for h in exp_labels]
    cluster_80 = [data['ood_cluster_80'][exp_labels[h]] for h in exp_labels]
    cluster_60 = [data['ood_cluster_60'][exp_labels[h]] for h in exp_labels]
    cluster_40 = [data['ood_cluster_40'][exp_labels[h]] for h in exp_labels]
    
    y_pos = np.arange(len(categories))  # y-axis positions for categories
    bar_width = 0.15  # Width of the bars
    colors = ['#CAC2F6', '#003F5C', '#4493A5', '#E67C69']  # Colors for each experiment
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for each experiment in each category
    for i, (experiment, color) in enumerate(zip(exp_labels, colors)):
        # Adjust the x-position of each experiment's bar
        bars = ax.bar(y_pos + (i - 1.5) * bar_width,  # Offset each bar by the appropriate amount
                      [cluster_99[i], cluster_80[i], cluster_60[i], cluster_40[i]], 
                      width=bar_width, color=color, label=experiment, align='center')

        # Add labels for each bar with value rounded to 3 decimals
        for j, bar in enumerate(bars):
            # Get the height of each bar (R2 value)
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.3f}', 
                    ha='center', va='bottom', rotation=90)

    # Set the ticks and labels for the x-axis (categories)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(categories_labels)
    ax.set_ylabel('R²')

    # Add legend outside the plot area
    ax.legend(title="Experiments", loc='upper left', bbox_to_anchor=(1.05, 1))

    if metric == 'R2': ax.set_xlabel('Clusters')
    elif metric == 'MAE': ax.set_xlabel('MAE')
    elif metric == 'p1mag': ax.set_xlabel(r'$p_1^{mag}$')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{os.path.basename(file_name)[:-4]}_vertical_with_labels.png', dpi=300)

# plot_r2_results('km', args.output_file[:-4]+'_R2.csv', 'R2')
# plot_r2_results('ki', args.output_file[:-4]+'_R2.csv', 'R2')

# plot_r2_results(args.output_file[:-4]+'_MAE.csv', 'MAE')
# plot_r2_results(args.output_file[:-4]+'_p1mag.csv', 'p1mag')

# plot_r2_vertical_results(args.output_file[:-4]+'_R2.csv', 'R2')
# plot_r2_vertical_results(args.output_file[:-4]+'_MAE.csv', 'MAE')
# plot_r2_vertical_results(args.output_file[:-4]+'_p1mag.csv', 'p1mag')