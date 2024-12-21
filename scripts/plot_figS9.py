import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parameters = ['kcat','km','ki']

for parameter, figlabel in zip(parameters, ['a','b','c']):
    labelcol = 'log10kcat_max' if parameter == 'kcat' else f'log10{parameter}_mean'
    # Load the data
    csv_file = f'../data/results/reproduce_results/{parameter}/{parameter}_uncertainty_preds_S9.csv'
    df = pd.read_csv(csv_file)

    unc_col = f'{labelcol}_mve_uncal_var'
    model_cols = [col for col in df.columns if col.startswith(labelcol) and 'model_' in col]

    unc = df[unc_col]

    prediction = df[labelcol]

    model_out = df[labelcol]
    epi_unc = np.var(df[model_cols], axis=1)
    alea_unc = unc - epi_unc

    # Specify column names for x and y axes
    x_col = 'Aleatoric_uncertainty' 
    y_col = 'Epistemic_uncertainty' 

    # Extract the x and y values
    x = alea_unc
    y = epi_unc

    # Create the scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.5)  # Set alpha for transparency
    plt.xscale('log')
    plt.yscale('log')

    # Plot the y=x line
    x_vals = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    plt.plot(x_vals, x_vals, 'r', label='y=x')

    # Calculate the fraction of points where x > y
    fraction_x_greater_y = np.sum(x > y) / len(x)

    # Add a text box with the fraction in the plot
    textstr = f'Fraction of points (x > y): {fraction_x_greater_y:.2f}'


    # Position the text box in the bottom-right corner of the plot (inside the plot area)
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.5))


    # Add labels and title
    plt.xlabel('Aleatoric uncertainty')
    plt.ylabel('Epistemic uncertainty')
    plt.title(f'Uncertainty Scatter Plot for held-out test predictions - {parameter}')

    # Show plot
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../results/figS9{figlabel}.png')