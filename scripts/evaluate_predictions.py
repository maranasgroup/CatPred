import sys
import os
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import spearmanr 

dir_prefix = sys.argv[2]
PARAMETER = sys.argv[1]
PREDS_DIR = f'../experiments/{PARAMETER}/test/{dir_prefix}'
DATA_DIR = '../CatPred-DB/data/processed/splits_wpdbs/'

TRAINVAL_MEANS = {'kcat': 0.96224, 'km': -0.72606 ,'ki': -1.84344}
PREDFILE_PREFIX = 'test_preds_unc_evi_mvewt_' #seq80.csv
DATAFILE_PREFIX = f'{PARAMETER}-random_test_sequence_' #80cluster.csv

TARGETCOL = f'log10{PARAMETER}_max' if PARAMETER=='kcat' else f'log10{PARAMETER}_mean'
STDEVCOL = f'{TARGETCOL}_mve_uncal_var'#f'{TARGETCOL}_evidential_total_mve_weighting_stdev' 
# STDEVCOL = f'{TARGETCOL}_evidential_total_uncal_var'#f'{TARGETCOL}_evidential_total_mve_weighting_stdev' 

SMILESCOL = 'reactant_smiles' if PARAMETER=='kcat' else 'substrate_smiles'

RANGE = [40,60,80,99]
import ipdb

def _error_calc(target, pred):
    errors = np.abs(np.array(target)-np.array(pred))
    # ipdb.set_trace()
    bins = np.arange(0, max(errors) + 0.1, 0.1)
    freqs,bin_edges = np.histogram(errors, bins)
    percs = 100*freqs/len(errors)
    cum_percs = np.cumsum(percs)
    
    #ipdb.set_trace()
    try:
        index_err1 = np.where(bin_edges==1.)[0][0]
        cum_perc_err1 = cum_percs[index_err1]
    except:
        cum_perc_err1 = None
    index_err1 = np.where(bin_edges==1.)[0][0]
    cum_perc_err1 = cum_percs[index_err1]

    bin_edges = bin_edges[1:]
    return cum_perc_err1, bin_edges, cum_percs

def _bin_by_std(target, pred, std, cutoff):
    df = pd.DataFrame({'target':target, 'pred': pred, 'stdev': std})
    return df[df.stdev<=cutoff]

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, pearsonr

def plot_corr_errors(x, y, savename='temp.pdf'):
    # Calculate 2D histogram
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, density=True)

    # Get density values for each point
    x_idx = np.clip(np.digitize(x, xedges) - 1, 0, len(xedges) - 2)
    y_idx = np.clip(np.digitize(y, yedges) - 1, 0, len(yedges) - 2)
    densities = heatmap[x_idx, y_idx]

    # Perform linear regression
    slope, intercept, _, _, _ = linregress(x, y)
    slope = 1
    intercept = 0
    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = slope * line_x + intercept

    # Create scatter plot with colored points using 'magma' colormap
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=densities, cmap='magma')
    plt.colorbar(label='Density')

    # Add the regression line with forest green color
    plt.plot(line_x, line_y, color='forestgreen', label='Linear fit')

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.title('Scatter plot with points colored by density and linear fit')
    plt.legend()

    # Save the figure as a PDF
    plt.savefig(savename)

def _calc_metrics(target, pred, std, R):
    # ipdb.set_trace()
    quartiles = np.percentile(std, [5, 25, 50, 75, 100])
    std_bins = list(quartiles)#[0.5,1.0,1.5,2.0,2.5,10]
    cum_perc_err1 = {}
    metrics_std = {'r2':{},'mae':{},'mse':{}}
    target_linear = np.power(10, target)
    pred_linear = np.power(10, pred)
    for bin in std_bins:
        df = _bin_by_std(target, pred, std, bin)
        target_, pred_ = df.target, df.pred
        print(len(target_), len(pred_))
        cum_perc_err1[bin], bins, cums = _error_calc(target_, pred_)
        metrics_std['r2'][bin] = r2_score(target_, pred_)
        metrics_std['mae'][bin] = mean_absolute_error(target_,pred_)
        metrics_std['mse'][bin] = mean_squared_error(target_,pred_)
    metrics_std['cum_perc_err1'] = cum_perc_err1
    plot_corr_errors(std, np.abs(target-pred), f'std-err-plot_{R}.pdf')
    return {'r2': r2_score(target, pred),
           'mae': mean_absolute_error(target,pred),
           'mse': mean_squared_error(target,pred), 
            'rho-err-std': spearmanr(np.abs(target-pred), std,alternative='two-sided'),
            'r-err-std': pearsonr(np.abs(target-pred), std),
            'r2_linear': r2_score(target_linear, pred_linear),
            'mae_linear': mean_absolute_error(target_linear, pred_linear),
            'mse_linear': mean_squared_error(target_linear, pred_linear)}, metrics_std

color1 = 'rgba(203, 101, 95, 0.8)'
color2 = 'rgba(92, 143, 198, 0.8)'
color3 = 'rgba(226, 192, 93, 0.8)'

if PARAMETER=='kcat': color = color1
elif PARAMETER=='km': color = color2
else: color = color3

def make_boxplot(percentile_stds, percentile_mae_avg, plot_outname, binwidth, color):
    # Binning y values by x values with step ranges starting from 0 in intervals of 0.5
    bins = np.arange(0, max(percentile_stds) + binwidth + binwidth/5, binwidth)  # Adjust range calculation
    # print(bins)
    digitized = np.digitize(percentile_stds, bins)
    #print(digitized)
    binned_data = {i: [] for i in range(1, len(bins))}  # Extend range to include the first box
    
    for i, val in enumerate(digitized):
        binned_data[val].append(percentile_mae_avg[i])

    #print(binned_data)
    
    # Create layout for aesthetics
    layout = go.Layout(
        xaxis=dict(
            tickfont=dict(size=18, color='black', family='Arial'),  # Font settings for x-axis ticks
            linecolor='black',  # Black-colored X-axis line
            tickvals=list(range(len(bins) + 1)),  # Set custom tick values for the bins
            ticktext=[f'{bins[i] if i < len(bins) else bins[i - 1] + binwidth:.1f}' for i in range(1, len(bins))],  # Format tick labels as intervals
            # tickangle=-45,  # Rotate x-axis tick labels anticlockwise
            tickwidth=1.25,  # Set tick width for x-axis
            ticks='outside',  # Place x-axis ticks outside the plot area
        ),
        yaxis=dict(
            tickfont=dict(size=18, color='black', family='Arial'),
            linecolor='black',  # Black-colored Y-axis line
            tickwidth=1.25,  # Set tick width for y-axis
            ticks='outside',  # Place y-axis ticks outside the plot area
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    )

    # Create boxplot data using the binned data
    boxplot_data = []

    for key, values in binned_data.items():
        if key < len(bins):  # Check to prevent index out of bounds
            boxplot_data.append(go.Box(
                y=values,
                name=f'{bins[key] if key < len(bins) else bins[key - 1] + binwidth:.2f}',  # Define label for each box as interval
                fillcolor=color,  # Set box fill color as transparent
                line=dict(color='black', width=1.25),  # Set outline color of each box to black with a thicker width
                boxpoints='outliers',  # Remove outlier points
            ))

    # Create the figure using the provided snippet and layout
    fig = go.Figure(data=boxplot_data, layout=layout)

    # Save the boxplot as SVG file
    pio.write_image(fig, plot_outname)
    
def _calc_metrics_unc(errors, stds, param, binwidth=0.5):
    percentiles = np.arange(1,99,0.1)
    stds, errors = zip(*sorted(zip(stds, errors)))
    bins = np.arange(0, max(stds) + binwidth, binwidth)  # Adjust range calculation
    # print(bins)
    digitized = np.digitize(stds, bins)
    percentile_stds = []
    for perc in percentiles:
        percentile_stds.append(np.percentile(stds, perc))

    percentile_mae_avg = []
    percentile_mae_std = []

    for perc in percentile_stds:
        items = []
        for err, std in zip(errors, stds):
            if std<=perc: 
                items.append(err)
        percentile_mae_avg.append(np.average(items))
        percentile_mae_std.append(np.std(items))
        
    make_boxplot(percentile_stds, percentile_mae_avg, f'{param}_unc_boxplot.svg', 0.8, color)

for R in RANGE:
    if R==0:
        PREDFILE_PREFIX2 = PREDFILE_PREFIX[:-1]
        DATAFILE_PREFIX2 = f'{PARAMETER}-random_test'
        datafile = f'{DATA_DIR}/{DATAFILE_PREFIX2}.csv'
        predsfile = f'{PREDS_DIR}/{PREDFILE_PREFIX2}.csv'
    else:
        datafile = f'{DATA_DIR}/{DATAFILE_PREFIX}{R}cluster.csv'
        predsfile = f'{PREDS_DIR}/{PREDFILE_PREFIX}seq{R}.csv'
        
    data_df = pd.read_csv(datafile)
    data_df.index = data_df[SMILESCOL] + data_df['sequence']
    preds_df = pd.read_csv(predsfile)
    preds_df.index = preds_df[SMILESCOL] + preds_df['sequence']
    pred = preds_df[TARGETCOL]
    target = [data_df.loc[ind][TARGETCOL] for ind in preds_df.index]
    f = open(f'{PARAMETER}_{R}_scatter.csv','w')
    for t, p in zip(target, pred):
        f.write(f'{t},{p}\n')
    f.close()
    print(preds_df.columns)
    std = preds_df[STDEVCOL]
    print('-'*50)
    print('Cutoff:', R)
    metrics, metrics_std = _calc_metrics(target,pred,std, R)
    print('-'*50)
    print('Naive mae with mean from training:')
    print(mean_squared_error(target, [TRAINVAL_MEANS[PARAMETER]]*len(target)))
    print('-'*50)
    for metric in metrics:
        print(metric, metrics[metric])
    print('-'*50)
    for metric in metrics_std:
        # if not metric in metrics_std: continue
        print(metric)
        for bin in metrics_std[metric]:
            print(bin, metrics_std[metric][bin])
        
    if R==0: _calc_metrics_unc(np.abs(target-pred),std, PARAMETER, 0.5)
    # break

# for R in RANGE:
#     if R==0:
#         PREDFILE_PREFIX2 = PREDFILE_PREFIX[:-1]
#         DATAFILE_PREFIX2 = f'{PARAMETER}-random_test'
#         datafile = f'{DATA_DIR}/{DATAFILE_PREFIX2}.csv'
#         predsfile = f'{PREDS_DIR}/{PREDFILE_PREFIX2}.csv'
#         break
 