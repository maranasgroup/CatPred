import os
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
import rdkit
import re
import matplotlib.pyplot as plt
import ete3
import re
import seaborn as sns

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

import sys
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "--data_dir", help="data directory", type=str, required=True)
    parser.add_argument("--out_dir", "--out_dir", help="output directory", type=str, required=True)
    parser.add_argument("--par", "--par", help="parameter to predict", 
                        type=str, required=True)
    parser.add_argument("--json", "--json", help="json brenda path", 
                        type=str, required=True)

    args, unparsed = parser.parse_known_args()
    parser = argparse.ArgumentParser()

    return args

args = parse_args()

param = args.par.upper()
json_name = args.json
out_path = args.out_dir
data_dir = args.data_dir

DATA_DIR = data_dir
PARAMETER = param

if PARAMETER=='KM':
    parameter_data = 'km_value'
elif PARAMETER=='KCAT':
    parameter_data = 'turnover_number'
elif PARAMETER=='KCATKM':
    parameter_data = 'kcat_km'
elif PARAMETER=='KI':
    parameter_data = 'ki_value'

OUTPUTNAME = f'./{out_path}/catpred-db_{PARAMETER}_train.csv'
OUTPUTNAME = f'./{out_path}/catpred-db_{PARAMETER}_train.csv'
OUTPUTNAME_SEQ = f'./{out_path}/catpred-db-seq_{PARAMETER}_train.csv'

# Get NCBI Taxonomy parser - to convert organism names into their taxonomic lineages
ncbi = ete3.NCBITaxa()
get_taxid_from_organism = lambda organism: ncbi.get_name_translator([organism])[organism][0]
org_to_taxid = json.load(open(f'{DATA_DIR}/organism_to_taxid.json'))

# Load BRENDA raw data
json_data = json.load(open(f'{json_name}'))
data = json_data['data']

# Create an empty dataframe and columns
df_kinetic = pd.DataFrame()
eccol = []
kcatcol = []
rxncol = []
kmcol = []
subcol = []
orgcol = []
unicol = []
commentcol = []
data_count = len(data)
nonsubstrate_col = []

for ec in data:
    if not parameter_data in data[ec] or not ('proteins' in data[ec] or 'organisms' in data[ec]):
        data_count-=1
        continue
    
    non_substrates = [] # list of cofactors and ions to ignore
    for sub_type in ['cofactor','metal_ions']:
        if sub_type in data[ec]:
            for sub in data[ec][sub_type]:
                non_substrates.append(sub['value'])
    
    non_substrates = set(non_substrates)
    
    rxn_by_org = {} # list of educts, products for each protein, org pair
    
    if 'organisms' in data[ec]: orgs_ec = data[ec]['organisms']
    else: orgs_ec = {}
    if 'proteins' in data[ec]: orgs_protein = data[ec]['proteins']
    else: orgs_protein = orgs_ec
    
    inhibitors = []
    all_natural = []
    
    if 'inhibitor' in data[ec]:
        for inh in data[ec]['inhibitor']:
            if 'value' in inh:
                inhibitors.append(inh['value'])
    if 'natural_reaction' in data[ec]: 
        for rxn in data[ec]['reaction']:
            try:
                reacs = rxn['educts']
                prods = rxn['products']
                all_natural.extend(reacs)
                all_natural.extend(prods)
            except KeyError:
                reacs = []
                prods = []
            if 'organisms' in rxn:
                orgs = rxn['organisms']
            elif 'proteins' in rxn:
                orgs = rxn['proteins']
            for org in orgs:
                if org in rxn_by_org: rxn_by_org[org].append((reacs,prods))
                else: rxn_by_org[org] = [(reacs,prods)]
        
    for entry in data[ec][parameter_data]:
        if 'num_value' in entry and 'value' in entry and 'organisms' in entry:
            kcat = entry['num_value']
            if 'organisms' in entry:
                org_now = entry['organisms']
                prot_now = org_now
            elif 'proteins' in entry:
                prot_now = entry['proteins']
                
            rxns = []
            
            for org in orgs:
                if org in rxn_by_org:
                    rxns = rxn_by_org[org]
                    
            rxn_subs = set()
            for rxn in rxns:
                for r in rxn[0]: rxn_subs.add(r)
                for p in rxn[0]: rxn_subs.add(p)
                        
            subname = entry['value']
            if PARAMETER=='KI':
                if not subname in all_natural+inhibitors:
                    continue
            else:
                if not subname in all_natural:
                    continue
            
            if not pd.isna(kcat) and not pd.isna(subname) and rxns:
                try:
                    kcat = float(kcat)
                except ValueError:
                    continue
                # get orgname if present, skip entry otherwise
                try:
                    orgname = data[ec]['organisms'][orgs[0]]['value']
                except KeyError:
                    continue
                # get uniprot accessions , organism names
                
                unis = []
                orgnames = []
                
                for org in prot_now:
                    # print(org)
                    if org in orgs_protein:
                        for each in orgs_protein[org]:
                            if 'accessions' in each:
                                unis.append(each['accessions'])
                            else:
                                unis.append(None)
                
                for org in org_now:
                    if org in orgs_ec:
                        if 'value' in orgs_ec[org]: 
                            orgnames.append(orgs_ec[org]['value'])
                        else:
                            orgnames.append(None)
                
                orgcol.append(orgnames[0])
                unis_ = []
                for u in unis:
                    if u is None: continue
                    unis_.append(u)
                
                if not unis_: unis_ = None
                    
                unicol.append(unis_)
                kcatcol.append(kcat)
                subcol.append(subname)
                eccol.append(ec)
                rxncol.append(rxns)
                
                if 'comment' in entry:
                    commentcol.append(entry['comment'])
                else:
                    commentcol.append('')
                    
df_kinetic[PARAMETER] = kcatcol
df_kinetic['SUBSTRATE'] = subcol
# df_kinetic['NON_SUBSTRATE'] = nonsubstrate_col
df_kinetic['Organism'] = orgcol
df_kinetic['COMMENT'] = commentcol
df_kinetic['UNIPROT'] = unicol
df_kinetic['EC'] = eccol
df_kinetic['REACTIONS'] = rxncol

taxid_col = []
for ind, row in df_kinetic.iterrows():
    org = row.Organism
    if not org in org_to_taxid: 
        taxid_col.append(None)
    else: 
        taxid_col.append(org_to_taxid[org])
df_kinetic['TAXONOMY_ID'] = taxid_col

# created using BRENDA ligand database
name_to_smiles = json.load(open(f'{DATA_DIR}/substrate_name_to_smiles_VERIFIED.json'))

smiles_col = []
for ind, row in df_kinetic.iterrows():
    sub = row.SUBSTRATE
    if not sub in name_to_smiles: smiles_col.append(None)
    else: smiles_col.append(name_to_smiles[sub])
df_kinetic['SMILES'] = smiles_col
df_kinetic.dropna(subset=['EC','TAXONOMY_ID','SMILES'],inplace=True)

#Get enzyme type and remove mutants and recombinants:
df_kinetic["ENZYME_TYPE"] = np.nan
df_kinetic.loc[pd.isnull(df_kinetic["COMMENT"])] = ""
df_kinetic["ENZYME_TYPE"] = "wild type"
df_kinetic["ENZYME_TYPE"][df_kinetic['COMMENT'].str.contains("mutant")] = "mutant"
df_kinetic["ENZYME_TYPE"][df_kinetic['COMMENT'].str.contains("presence of")] = "regulated"
df_kinetic["ENZYME_TYPE"][df_kinetic['COMMENT'].str.contains("mutate")] = "mutant"
df_kinetic["ENZYME_TYPE"][df_kinetic['COMMENT'].str.contains("chimera")] = "mutant"
df_kinetic["ENZYME_TYPE"][df_kinetic['COMMENT'].str.contains("recombinant")] = "recombinant"
df_kinetic["ENZYME_TYPE"][df_kinetic['COMMENT'].str.contains("allozyme")] = "allozyme"
df_kinetic["ENZYME_TYPE"][df_kinetic['COMMENT'].str.contains("alloenzyme")] = "allozyme"
df_kinetic["ENZYME_TYPE"][df_kinetic['COMMENT'].str.contains("isozyme")] = "isozyme"
df_kinetic["ENZYME_TYPE"][df_kinetic['COMMENT'].str.contains("isoenzyme")] = "isozyme"
df_kinetic["ENZYME_TYPE"][df_kinetic['COMMENT'].str.contains("isoform")] = "isozyme"

phrow = []
temprow = []
for com in df_kinetic.COMMENT:
    pat = r'[0-9][0-9].C'
    try:
        temp = int(re.findall(pattern=pat, string=com)[0].split('C')[0][:2])
        pat = r'pH [0-9]\.[0-9]'
        ph = float(re.findall(pattern=pat, string=com)[0].split('pH')[-1].strip())
    except:
        temp = None
        ph = None
    phrow.append(ph)
    temprow.append(temp)
    
df_kinetic['PH'] = phrow
df_kinetic['TEMPERATURE'] = temprow

df_final = df_kinetic[df_kinetic.ENZYME_TYPE=='wild type']

logkmrow = []
for km in df_final[PARAMETER]:
    logkmrow.append(np.log10(km))
df_final[f'LOG10_{PARAMETER}'] = logkmrow

df_final.reset_index(inplace=True, drop=True)

groups = df_final.groupby(['Organism', 'SMILES','EC'])

stdcol = []
meancol = []
mediancol = [] 
rangecol = [] #min to max

allcol = [] #all values as string
dfmean = pd.DataFrame()
for _, group in tqdm(groups):
    d = group.iloc[0:1].copy()
    if PARAMETER=='KCAT': d[f'target_{PARAMETER}'] = group[f'LOG10_{PARAMETER}'].max()
    elif PARAMETER=='KM' or PARAMETER=='KI': d[f'target_{PARAMETER}'] = group[f'LOG10_{PARAMETER}'].mean()
    elif PARAMETER=='KCATKM' or PARAMETER=='KI': d[f'target_{PARAMETER}'] = group[f'LOG10_{PARAMETER}'].max()
    dfmean = pd.concat([dfmean, d])

remove = []
for ind, row in tqdm(dfmean.iterrows()):
    smiles = row.SMILES
    try:
        mol = AllChem.MolFromSmiles(smiles)
        feature_list = AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=2048)
    except:
        remove.append(ind)
dfmean.drop(remove, inplace=True)
dfmean.reset_index(inplace=True)

from data_utils import featurize
_, df_feats = featurize(dfmean, PARAMETER, 
              root_path = ".", 
              data_dir = './data/', 
              redo_feats = True,
              baseline=False,
              add_esm=False,
              skip_embeds=False,
              skip_fp = True,
              fp_radius = 2, 
              fp_length = 2048,
              include_y = False)

ec_embed_cols = ["EC1", "EC2", "EC3", "EC"]
tax_embed_cols = [
"SUPERKINGDOM",
"PHYLUM",
"CLASS",
"ORDER",
"FAMILY",
"GENUS",
"SPECIES"]

towrite = df_feats[['EC','Organism','SUBSTRATE','SMILES',f'target_{PARAMETER}']+ec_embed_cols+tax_embed_cols]
towrite.to_csv(f'{OUTPUTNAME}')

ec_tax_pairs = []
for ind, row in tqdm(dfmean.iterrows()):
    unis = row.UNIPROT
    ec = row.EC
    tax = int(row.TAXONOMY_ID)
    if unis is None:
        if not (ec, tax) in ec_tax_pairs:
            ec_tax_pairs.append((ec,tax))
    else:
        continue

ec_tax_pairs_uni_dict = json.load(open(f'{DATA_DIR}/ec_tax_uniprot_dict.json'))
ec_tax_pairs_dict = {}
for each, out in ec_tax_pairs_uni_dict.items():
    pair = each.split('__')
    pair = pair[0], int(pair[1])
    ec_tax_pairs_dict[pair] = out


ec_tax_pairs_to_fetch = []
for pair in ec_tax_pairs:
    if not pair in ec_tax_pairs_dict: ec_tax_pairs_to_fetch.append(pair)

from joblib import delayed, Parallel
import requests

def get_fastas(pair):
    ecid, taxid = pair
    url = f"https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28%28taxonomy_id%3A{taxid}%29+AND+%28ec%3A{ecid}%29%29"
    fastas = None
    if not (ecid, taxid) in ec_tax_pairs_dict:
        r = requests.get(url)
        if r.status_code==200:
            fastas = r.text
    
    return fastas

outputs = Parallel(n_jobs=30, verbose=5)(delayed(get_fastas)(pair) for pair in ec_tax_pairs_to_fetch)

for pair, output in zip(ec_tax_pairs_to_fetch, outputs):
    ec_tax_pairs_dict[pair] = output


for pair, output in ec_tax_pairs_dict.items():
    if not output is None:
        if output: 
            ec_tax_pairs_uni_dict[pair[0]+'__'+str(pair[1])] = output



import json
f = open(f'{DATA_DIR}/ec_tax_uniprot_dict.json','w')
f.write(json.dumps(ec_tax_pairs_uni_dict, indent=True))
f.close()

unicol = []
for ind, row in tqdm(dfmean.iterrows()):
    unis = row.UNIPROT
    ec = row.EC
    tax = int(row.TAXONOMY_ID)
    ectax = ec + '__' + str(tax)
    if unis is None:
        if ectax in ec_tax_pairs_uni_dict:
            fastas = ec_tax_pairs_uni_dict[ectax]
            unis = []
            lines = fastas.split('\n')
            for each in lines:
                if each.startswith('>'):
                    unis.append(each.split('|')[1])
            # break
    elif unis:
        if len(unis)==0: 
            unicol.append(None)
            continue
        elif not unis[0]:
            if ectax in ec_tax_pairs_uni_dict:
                fastas = ec_tax_pairs_uni_dict[ectax]
                unis = []
                lines = fastas.split('\n')
                for each in lines:
                    if each.startswith('>'):
                        unis.append(each.split('|')[1])
    else:
        unis = None
        
    unicol.append(unis)

unicol2 = []
for u in unicol:
    if not u is None:
        if len(u)!=0:
            unicol2.append(u)
        else:
            unicol2.append(None)
    else:
        unicol2.append(u)

dfmean['UNIPROT'] = unicol2

dfmean_seq = dfmean.dropna(subset=['UNIPROT'],inplace=False)

dfmean_seq.reset_index(inplace=True, drop=True)


n_uniprot_col = []
for ind, row in dfmean_seq.iterrows():
    unis = np.array(row.UNIPROT).flatten()
    n_uniprot_col.append(len(unis))


dfmean_seq['N_UNIPROT'] = n_uniprot_col

x = dfmean_seq[dfmean_seq.N_UNIPROT!=1]

y = dfmean_seq[dfmean_seq.N_UNIPROT==1]

y.reset_index(inplace=True,drop=True)

unicol = []
for ind, row in tqdm(y.iterrows()):
    unicol.append(np.array(row.UNIPROT).flatten()[0])
y['UNIPROT'] = unicol


seqcol = []
uniset = y.UNIPROT.unique()
def _get_sequence(uni): 
    r = requests.get(f'https://rest.uniprot.org/uniprotkb/{uni}.fasta')
    if r.status_code==200:
        lines = r.text.split('\n')
        seq = ''.join(lines[1:])
        return seq
outputs = Parallel(n_jobs=30, verbose=5)(delayed(_get_sequence)(uni) for uni in uniset)
uni_to_seq = {}
for uni, seq in zip(uniset, outputs):
    uni_to_seq[uni] = seq
    
seqrow = []
for ind, row in tqdm(y.iterrows()):
    if row.UNIPROT in uni_to_seq: 
        if uni_to_seq[row.UNIPROT].strip():
            seqrow.append(uni_to_seq[row.UNIPROT])
        else:
            seqrow.append(None)
    else: seqrow.append(None)
y['SEQUENCE'] = seqrow

y.dropna(subset=['SEQUENCE'], inplace=True)
y.reset_index(inplace=True, drop=True)
towrite = y[['EC','Organism','SMILES','UNIPROT','SEQUENCE', f'target_{PARAMETER}']]

towrite.to_csv(f'{OUTPUTNAME_SEQ}')
