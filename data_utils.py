import pandas as pd
import numpy as np
import time
import json
import sys
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import ete3

def load_vocabulary(parameter, root_path, data_dir):
    vocab_dic = json.load(open(f"{root_path}/{data_dir}/vocab/{parameter}_vocab.json"))
    return vocab_dic

def add_integer_embedding(vocab_dic, df, colname):
    dic = vocab_dic[colname]
    temp = []
    temp_vec = []
    for name in df[colname].astype("str"):
        if not name in dic: 
            name = 'UNK'
        temp.append(dic[name])
        temp_vec.append([dic[name]])
    
    df[f"{colname}_INTEGER"] = temp
    df[f"{colname}_INTEGER_VEC"] = temp_vec
    return df

def add_onehot_embedding(vocab_dic, df, colname):
    ints = df[f"{colname}_INTEGER"]
    dic = vocab_dic[colname]
    keys = []
    values = []
    for k, v in dic.items():
        values.append(int(v))
        keys.append(str(k))

    x = np.array(values).astype('int')
    onehot = np.zeros((len(df), len(x)))
    i = 0
    for each in ints:
        loc = values.index(each)
        onehot[i, loc] = 1
        i+=1

    onehot = list(onehot.astype('int'))
    df[f'{colname}_ONEHOT'] = onehot
    return df

def add_fps(df, radius=2, length=2048):
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
    ec_words = {f"EC{i}": ec_chars[:i] for i in range(1,4)}
    ec_words['EC'] = ec
    return ec_words

def get_tax_words(organism, ncbi, org_to_taxid, tax_embed_cols):
    get_taxid_from_organism = lambda organism: ncbi.get_name_translator([organism])[organism][0]
    try:
        taxid = get_taxid_from_organism(organism)
    except:
        if not organism in org_to_taxid:
            taxid = None
            print(f'Organism {organism} not found in NCBI or CatPred database! Making predictions using UNK words, this may lead to inaccurate predictions')
        else:
            taxid = org_to_taxid[organism]
            
        return {tax: 'UNK' for tax in tax_embed_cols}
    
    lineage = ncbi.get_lineage(taxid)
    rank_dict = ncbi.get_rank(lineage)
    rank_dict_return = {}
    for rankid, rankname in rank_dict.items():
        if rankname.upper() in tax_embed_cols: rank_dict_return[rankname.upper()] = ncbi.get_taxid_translator([rankid])[rankid]
        
    return rank_dict_return
    
def featurize(df, parameter, root_path = ".", data_dir = './data/', include_y = False):
    ncbi = ete3.NCBITaxa(taxdump_file=f'{root_path}/{data_dir}/taxdump.tar.gz')
    org_to_taxid = json.load(open('./data/organism_to_taxid.json'))
    
    ec_embed_cols = ["EC1", "EC2", "EC3", "EC"]
    tax_embed_cols = [
        "SUPERKINGDOM",
        "PHYLUM",
        "CLASS",
        "ORDER",
        "FAMILY",
        "GENUS",
        "SPECIES",
    ]

    print("Preparing Data ...")

    print("Adding EC and TC words from pre-defined vocabulary ...")
    ec_words = []
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

    vocab_dic = load_vocabulary(parameter, root_path, data_dir)

    for EC in ec_embed_cols:
        df = add_integer_embedding(vocab_dic, df, EC)
    for TAX in tax_embed_cols:
        df = add_integer_embedding(vocab_dic, df, TAX)
        #add onehots
    for EC in ec_embed_cols:
        df = add_onehot_embedding(vocab_dic, df, EC)
    for TAX in tax_embed_cols:
        df = add_onehot_embedding(vocab_dic, df, TAX)
    
    print("Adding substrate fingerprints ...")
    df = add_fps(df)
    
    # by default these should be there
    features_to_add = [df[['FP']]]
    embed_type = 'ONEHOT'

    for each in ec_embed_cols+tax_embed_cols:
        features_to_add.append(df[[f"{each}_{embed_type}"]])

    # total minus default ones
    n_feats = len(features_to_add)

    prepared_df = pd.concat(features_to_add, axis=1)

    X_vals = prepared_df.iloc[:,:].values # only feats
    
    Xs = []
    for i in range(n_feats):
        Xs.append(np.stack(X_vals[:,i]))
            
    X = np.concatenate(Xs, axis=-1)

    if include_y: 
        try:
            y = np.array(df[f'LOG10_{parameter}_MEDIAN'].values)
            return (X,y), df
        except:
            return X, df
    else: 
        return X, df