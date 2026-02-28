"""
Enzyme Kinetics Parameter Prediction Script

This script predicts enzyme kinetics parameters (kcat, Km, or Ki) using a pre-trained model.
It processes input data, generates predictions, and saves the results.

Usage:
    python demo_run.py --parameter <kcat|km|ki> --input_file <path_to_input_csv> --checkpoint_dir <path_to_pretrained_checkpoint_dir> [--use_gpu]

Dependencies:
    pandas, numpy, rdkit, IPython, argparse
"""

import time
import os
import subprocess
import pandas as pd
import numpy as np
from IPython.display import Image, display
from rdkit import Chem
from IPython.display import display, Latex, Math
import argparse


def prepare_prediction_inputs(parameter, input_file_path):
    df = pd.read_csv(input_file_path)
    required_columns = {"SMILES", "sequence", "pdbpath"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        print(
            f'Missing required column(s) in input file: {", ".join(sorted(missing_columns))}.'
        )
        return None

    conflicting_pdbpaths = (
        df.groupby("pdbpath")["sequence"]
        .nunique(dropna=False)
        .loc[lambda value: value > 1]
    )
    if len(conflicting_pdbpaths) > 0:
        preview = ", ".join(conflicting_pdbpaths.index.astype(str).tolist()[:5])
        print(
            "Found pdbpath values mapped to multiple sequences. "
            f"Each unique sequence must have a unique pdbpath. Examples: {preview}"
        )
        return None

    smiles_list = df.SMILES
    seq_list = df.sequence
    smiles_list_new = []

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

    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    for i, seq in enumerate(seq_list):
        if not set(seq).issubset(valid_aas):
            print(f'Invalid Enzyme sequence input in row {i}!')
            print('Correct your input! Exiting..')
            return None

    input_file_base, _ = os.path.splitext(input_file_path)
    input_file_new_path = f'{input_file_base}_input.csv'
    df['SMILES'] = smiles_list_new
    df.to_csv(input_file_new_path, index=False)

    test_file_prefix = input_file_new_path[:-4]
    return {
        "input_csv": input_file_new_path,
        "records_file": f"{test_file_prefix}.json.gz",
        "output_csv": f"{test_file_prefix}_output.csv",
    }

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

    missing_cols = [col for col in [target_col, unc_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f'Prediction output is missing required column(s): {", ".join(missing_cols)}'
        )

    for _, row in df.iterrows():
        model_cols = [col for col in row.index if col.startswith(target_col) and 'model_' in col]

        unc = row[unc_col]
        prediction = row[target_col]
        prediction_linear = np.power(10, prediction)

        if model_cols:
            model_outs = np.array([row[col] for col in model_cols])
            epi_unc = np.var(model_outs)
        else:
            epi_unc = 0.0
        alea_unc = max(unc - epi_unc, 0.0)
        epi_unc = np.sqrt(epi_unc)
        alea_unc = np.sqrt(alea_unc)
        unc = np.sqrt(max(unc, 0.0))

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
    run_paths = prepare_prediction_inputs(args.parameter, args.input_file)
    if run_paths is None:
        return

    outfile = run_paths["output_csv"]
    print('Predicting.. This will take a while..')

    env = os.environ.copy()
    env["PROTEIN_EMBED_USE_CPU"] = "0" if args.use_gpu else "1"

    create_records_cmd = [
        "python",
        "./scripts/create_pdbrecords.py",
        "--data_file",
        run_paths["input_csv"],
        "--out_file",
        run_paths["records_file"],
    ]
    predict_cmd = [
        "python",
        "predict.py",
        "--test_path",
        run_paths["input_csv"],
        "--preds_path",
        outfile,
        "--checkpoint_dir",
        args.checkpoint_dir,
        "--uncertainty_method",
        "mve",
        "--smiles_column",
        "SMILES",
        "--individual_ensemble_predictions",
        "--protein_records_path",
        run_paths["records_file"],
    ]

    create_records_result = subprocess.run(create_records_cmd, env=env)
    if create_records_result.returncode != 0:
        print(
            f"Protein record generation failed with exit code {create_records_result.returncode}."
        )
        return

    predict_result = subprocess.run(predict_cmd, env=env)
    if predict_result.returncode != 0:
        print(f"Prediction command failed with exit code {predict_result.returncode}.")
        return

    if not os.path.exists(outfile):
        print(f'Prediction output file was not generated: {outfile}')
        return

    output_final = get_predictions(args.parameter, outfile)
    filename = outfile.split('/')[-1]
    os.makedirs('../results', exist_ok=True)
    output_final.to_csv(f'../results/{filename}', index=False)
    print('Output saved to results/', filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict enzyme kinetics parameters.")
    parser.add_argument("--parameter", type=str, choices=["kcat", "km", "ki"], required=True,
                        help="Kinetics parameter to predict (kcat, km, or ki)")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input CSV file")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for prediction (default is CPU)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to the model checkpoint directory")

    args = parser.parse_args()
    args.parameter = args.parameter.lower()

    main(args)
