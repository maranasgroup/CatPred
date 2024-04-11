parameter = 'kcat' # allowed values: ["kcat", "Km", "Ki"] 
parameter = parameter.lower()

use_cpu = 1 # set to 0 if you have GPU enabled

input_file_path = '../data/demo/batch_kcat.csv'

import time
import os
import pandas as pd
import numpy as np
from IPython.display import Image, display
from rdkit import Chem
from IPython.display import display, Latex, Math

print(os.getcwd())

def create_csv_sh(parameter, input_file_path):
    df = pd.read_csv(input_file_path)
    smiles_list = df.SMILES
    seq_list = df.sequence
    smiles_list_new = []
    i=0
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            smi = Chem.MolToSmiles(mol)
        except:
            print(f'Invalid SMILES input in input row {i}')
            print('Correct your input! Exiting..')
            return
        if parameter=='kcat':
            if '.' in smi:
              x = smi.split('.')
              y = sorted(x)
              smi = '.'.join(y)
        smiles_list_new.append(smi)
        i+=1
  
    i=0
    valid_aas = list('ACDEFGHIKLMNPQRSTVWY')
    for seq in seq_list:
      for aa in seq:
        if not aa in valid_aas:
          print(f'Invalid Enzyme sequence input in row {i}!')
          print('Correct your input! Exiting..')
          return
      i+=1

    input_file_new_path = f'{input_file_path[:-4]}_input.csv'
    df['SMILES'] = smiles_list_new
    df.to_csv(input_file_new_path)
    
    f = open(f'predict.sh', 'w')
    f.write(f'''
    TEST_FILE_PREFIX={input_file_new_path[:-4]}
    RECORDS_FILE=${{TEST_FILE_PREFIX}}.json
    CHECKPOINT_DIR=./production_models/{parameter}/
    
    python ./scripts/create_pdbrecords.py --data_file ${{TEST_FILE_PREFIX}}.csv --out_file ${{RECORDS_FILE}}
    python predict.py --test_path ${{TEST_FILE_PREFIX}}.csv --preds_path ${{TEST_FILE_PREFIX}}_output.csv --checkpoint_dir $CHECKPOINT_DIR --uncertainty_method mve --smiles_column SMILES --individual_ensemble_predictions --protein_records_path $RECORDS_FILE
    ''')
    f.close()
    
    return input_file_new_path[:-4]+'_output.csv'

outfile = create_csv_sh(parameter, input_file_path)

print('Predicting.. This will take a while..\n')

if use_cpu:
    os.system("export PROTEIN_EMBED_USE_CPU=1;./predict.sh")
else:
    os.system("export PROTEIN_EMBED_USE_CPU=0;./predict.sh")

def get_predictions(parameter, outfile):
    df = pd.read_csv(outfile)
    pred_col = []
    pred_logcol = []
    pred_sd_totcol = []
    pred_sd_aleacol = []
    pred_sd_epicol = []
    
    for ind, row in df.iterrows():
        unit = 'mM'
        if parameter=='kcat':
            parameter_print = 'k_{cat}'
            parameter_print_log = 'log_{10}(k_{cat})'
            target_col = 'log10kcat_max'
            unit = 's^(-1)'
        elif parameter=='km':
            target_col = 'log10km_mean'
            parameter_print = 'K_{m}'
            parameter_print_log = 'log_{10}(K_{m})'
        else:
            target_col = 'log10ki_mean'
            parameter_print = 'K_{i}'
            parameter_print_log = 'log_{10}(K_{i})'
    
        unc_col = f'{target_col}_mve_uncal_var'
        model_cols = [col for col in row.columns if col.startswith(target_col) and 'model_' in col]
    
        unc = row[unc_col].iloc[0]
    
        prediction = row[target_col].iloc[0]
        prediction_linear = np.power(10, prediction)
    
        model_out = row[target_col].iloc[0]
        model_outs = np.array([row[col].iloc[0] for col in model_cols])
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

output_final = get_predictions(parameter, outfile)
filename = outfile.split('/')[-1]
output_final.to_csv(f'../results/{filename}')
print('Output saved to results/', filename)