import pandas as pd
import argparse
from data_utils import featurize
from skops.io import load
import numpy as np
from catboost import CatBoostRegressor, Pool
import ete3
import json
import sys
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input csv file", type=str, required=True)
    parser.add_argument("-par", "--parameter", help="parameter to predict", 
                        type=str, required=True)

    args, unparsed = parser.parse_known_args()
    parser = argparse.ArgumentParser()

    return args

args = parse_args()

def add_fps(df, radius=3, length=2048):
    failed = []
    fps = []
    for i, row in tqdm(df.iterrows()):
        smi = row["SMILES"]
        try:
            mol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol,radius,length)
            fps.append(np.array(fp))
        except:
            failed.append(i)
            fps.append(None)
            
    df["FP"] = fps
    df.drop(failed, inplace=True)
    return df
    
def get_ec_words(ec):
    if '-' in ec: ec.replace('-','UNK')
    ec_chars = ec.split('.')
    ec_words = {f"EC{i}": '.'.join(ec_chars[:i]) for i in range(1,4)}
    ec_words['EC'] = ec
    return ec_words

def get_tax_words(organism, ncbi, org_to_taxid, tax_embed_cols):
    get_taxid_from_organism = lambda organism: ncbi.get_name_translator([organism])[organism][0]
    try:
        taxid = get_taxid_from_organism(organism)
    except:
        if not organism in org_to_taxid:
            taxid = None
            print(f'Organism {organism} not found in NCBI or CatPred database! \
                Making predictions using UNK words, this may lead to inaccurate predictions')
        else:
            taxid = org_to_taxid[organism]
            
        return {tax: 'UNK' for tax in tax_embed_cols}
    
    lineage = ncbi.get_lineage(taxid)
    rank_dict = ncbi.get_rank(lineage)
    rank_dict_return = {}
    for rankid, rankname in rank_dict.items():
        if rankname.upper() in tax_embed_cols: rank_dict_return[rankname.upper()] = ncbi.get_taxid_translator([rankid])[rankid]
        
    return rank_dict_return

print(args)
root_path = '.'
data_dir = './data/'

dfin = pd.read_csv(args.input)

df = dfin.copy()

ncbi = ete3.NCBITaxa(taxdump_file=f'{root_path}/{data_dir}/taxdump.tar.gz', update=False)
org_to_taxid = json.load(open(f'./{root_path}/{data_dir}/organism_to_taxid.json'))

print("Preparing Data ...")

print("Adding EC and TC words from pre-defined vocabulary ...")
ec_words = []
ec_embed_cols = ["EC1", "EC2", "EC3", "EC"]
tax_embed_cols = [
"SUPERKINGDOM",
"PHYLUM",
"CLASS",
"ORDER",
"FAMILY",
"GENUS",
"SPECIES"]

for ind, row in df.iterrows():
    words = get_ec_words(row.EC)
    ec_words.append(words)
for col in ec_embed_cols:
    col_values = [ec_words[i][col] for i in range(len(df))]
    df[col] = col_values

tax_words = []
for ind, row in df.iterrows():
    words = get_tax_words(row.Organism, ncbi, org_to_taxid, tax_embed_cols)
    tax_words.append(words)
for col in tax_embed_cols:
    col_values = []
    for i in range(len(df)):
        if col in tax_words[i]:
            col_values.append(tax_words[i][col])
        else:
            col_values.append('UNK')
    df[col] = col_values

def add_fps(df, radius=3, length=2048):
    failed = []
    fps = []
    for i, row in tqdm(df.iterrows()):
        smi = row["SMILES"]
        try:
            mol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol,radius,length)
            fps.append(np.array(fp))
        except:
            failed.append(i)
            fps.append(None)
            
    df["FP"] = fps
    df.drop(failed, inplace=True)
    return df

df = add_fps(df)
fps_all = np.stack(df.FP)
fps_transp = np.transpose(fps_all)
for i in range(2048):
    df[f'FP_{i}'] = fps_transp[i]
df_test = df.copy()

other_cols = ['PH','TEMPERATURE']
sub_cols = [f'FP_{i}' for i in range(2048)]
for col in ec_embed_cols+tax_embed_cols:
    df_test[col] = df_test[col].astype('str')
    
model = CatBoostRegressor(n_estimators=2000,
                       loss_function='RMSE',
                       learning_rate=0.4,
                       depth=3, 
                       random_state=0,
                       verbose=False)

model.load_model(f'./models/catpred-{args.parameter.upper()}-production.json', format='json')
feat_cols = ec_embed_cols+tax_embed_cols+other_cols+sub_cols
remove = [col for col in df_test.columns if not col in feat_cols]
X_test = df_test.drop(columns=remove)

pool_test = Pool(X_test, cat_features=ec_embed_cols+tax_embed_cols)

print('Making predictions ...')
preds = model.predict(pool_test)

# raise to power 10
output_col = np.power(10, preds)

if args.parameter.upper()=='KCAT':
    outname = 'KCAT s^(-1)'
elif args.parameter.upper()=='KM':
    outname = 'KM mM'
if args.parameter.upper()=='KI':
    outname = 'KI mM'

dfin[outname] = output_col
dfin.to_csv(f'{args.input[:-4]}_result.csv')
print(f'Predictions saved to {args.input[:-4]}_result.csv')
