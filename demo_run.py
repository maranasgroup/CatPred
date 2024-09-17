"""
Enzyme Kinetics Parameter Prediction Script

This script predicts enzyme kinetics parameters (kcat, Km, or Ki) using a pre-trained model.
It processes input data, generates predictions, and saves the results.

Usage:
    python script_name.py --parameter <kcat|km|ki> --input_file <path_to_input_csv> [--use_gpu]

Dependencies:
    pandas, numpy, rdkit, IPython, argparse
"""

import time
import os
import pandas as pd
import numpy as np
from IPython.display import Image, display
from rdkit import Chem
from IPython.display import display, Latex, Math
import argparse

def create_csv_sh(parameter, input_file_path):
    """
    Process input data and create a shell script for prediction.

    Args:
        parameter (str): The kinetics parameter to predict.
        input_file_path (str): Path to the input CSV file.

    Returns:
        str: Path to the output CSV file.
    """
    df = pd.read_csv(input_file_path)
    smiles_list = df.SMILES
    seq_list = df.sequence
    smiles_list_new = []

    # Process SMILES strings
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            smi = Chem.MolToSmiles(mol)
            if parameter == 'kcat' and '.' in smi:
                smi = '.'.join(sorted(smi.split('.')))
            smiles_list_new.append(smi)
        except:
            print(f'Invalid SMILES input in input row {i}')
            print('Correct your input! Exiting..')
            return None

    # Validate enzyme sequences
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    for i, seq in enumerate(seq_list):
        if not set(seq).issubset(valid_aas):
            print(f'Invalid Enzyme sequence input in row {i}!')
            print('Correct your input! Exiting..')
            return None

    # Save processed input
    input_file_new_path = f'{input_file_path[:-4]}_input.csv'
    df['SMILES'] = smiles_list_new
    df.to_csv(input_file_new_path)

    # Create shell script for prediction
    with open('predict.sh', 'w') as f:
        f.write(f'''
        TEST_FILE_PREFIX={input_file_new_path[:-4]}
        RECORDS_FILE=${{TEST_FILE_PREFIX}}.json
        CHECKPOINT_DIR=./production_models/{parameter}/
        
        python ./scripts/create_pdbrecords.py --data_file ${{TEST_FILE_PREFIX}}.csv --out_file ${{RECORDS_FILE}}
        python predict.py --test_path ${{TEST_FILE_PREFIX}}.csv --preds_path ${{TEST_FILE_PREFIX}}_output.csv --checkpoint_dir $CHECKPOINT_DIR --uncertainty_method mve --smiles_column SMILES --individual_ensemble_predictions --protein_records_path $RECORDS_FILE
        ''')

    return input_file_new_path[:-4]+'_output.csv'

def get_predictions(parameter, outfile):
    """
    Process prediction results and add additional metrics.

    Args:
        parameter (str): The kinetics parameter that was predicted.
        outfile (str): Path to the output CSV file from the prediction.

    Returns:
        pandas.DataFrame: Processed predictions with additional metrics.
    """
    df = pd.read_csv(outfile)
    pred_col, pred_logcol, pred_sd_totcol, pred_sd_aleacol, pred_sd_epicol = [], [], [], [], []

    unit = 'mM'
    if parameter == 'kcat':
        target_col = 'log10kcat_max'
        unit = 's^(-1)'
    elif parameter == 'km':
        target_col = 'log10km_mean'
    else:
        target_col = 'log10ki_mean'

    unc_col = f'{target_col}_mve_uncal_var'
    
    for _, row in df.iterrows():
        model_cols = [col for col in row.index if col.startswith(target_col) and 'model_' in col]
        
        unc = row[unc_col]
        prediction = row[target_col]
        prediction_linear = np.power(10, prediction)
        
        model_outs = np.array([row[col] for col in model_cols])
        epi_unc = np.var(model_outs)
        alea_unc = unc - epi_unc
        epi_unc = np.sqrt(epi_unc)
        alea_unc = np.sqrt(alea_unc)
        unc = np.sqrt(unc)
        
        pred_col.append(prediction_linear)
        pred_logcol.append(prediction)
        pred_sd_totcol.append(unc)
        pred_sd_aleacol.append(alea_unc)
        pred_sd_epicol.append(epi_unc)

    df[f'Prediction_({unit})'] = pred_col
    df['Prediction_log10'] = pred_logcol
    df['SD_total'] = pred_sd_totcol
    df['SD_aleatoric'] = pred_sd_aleacol
    df['SD_epistemic'] = pred_sd_epicol

    return df

def main(args):
    """
    Main function to run the prediction process.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    print(os.getcwd())

    outfile = create_csv_sh(args.parameter, args.input_file)
    if outfile is None:
        return

    print('Predicting.. This will take a while..\n')

    if args.use_gpu:
        os.system("export PROTEIN_EMBED_USE_CPU=0;./predict.sh")
    else:
        os.system("export PROTEIN_EMBED_USE_CPU=1;./predict.sh")

    output_final = get_predictions(args.parameter, outfile)
    filename = outfile.split('/')[-1]
    output_final.to_csv(f'../results/{filename}')
    print('Output saved to results/', filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict enzyme kinetics parameters.")
    parser.add_argument("--parameter", type=str, choices=["kcat", "km", "ki"], required=True,
                        help="Kinetics parameter to predict (kcat, km, or ki)")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input CSV file")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for prediction (default is CPU)")

    args = parser.parse_args()
    args.parameter = args.parameter.lower()

    main(args)
