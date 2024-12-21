import numpy as np
import pandas as pd
from rdkit import Chem
import json

cofactors = pd.read_csv('cofactors.csv').KEGG_ID
cofactor_smiles = []

def get_smiles(keggid, kegg_molfile_path = './KEGG_substrate_mols/'):
    try:
        mol = Chem.MolFromMolFile(kegg_molfile_path+keggid+'.mol')
    except: 
        return ''
    smi = Chem.MolToSmiles(mol)
    return smi    

for each in cofactors:
    cofactor_smiles.append(get_smiles(each))

cofactor_smiles = set(cofactor_smiles)

for PARAMETER in ['kcat','km','ki']:
    for suffix in ['train','trainval','trainvaltest','val','test']:
        print(PARAMETER, suffix)
        DATA_PATH = f'/home/ubuntu/mychemprop/CatPred-DB/data/processed/splits_wpdbs/{PARAMETER}-random_{suffix}.csv'
        OUTNAME = f'/home/ubuntu/DLKcat/DeeplearningApproach/Code/model/{PARAMETER}-random_{suffix}_dlkcat.json'

        df = pd.read_csv(DATA_PATH)

        data = []
        smilist = set()

        for ind, row in df.iterrows():
            ec = row.ec
            org = row.taxonomy_id
            if PARAMETER=='kcat': 
                unit = 's^(-1)'
                val = np.power(10, row.log10kcat_max)
                smis = row.reactant_smiles.split('.')
                smi = None
                if len(smis)==1:
                    smi = smis[0]
                else:
                    for s in smis:
                        if not s in cofactor_smiles: 
                            smi = s
                    # if still not found
                    if smi is None:
                        smi = sorted(smis)[0]
            else:
                smi = row.substrate_smiles    
                val = np.power(10, row[f'log10{PARAMETER}_mean'])       
                unit = 'mM'

            seq = row.sequence
            entry = {'ECNumber': ec, 'Organism': str(int(org)), 'Smiles': smi, 'Sequence': seq, 
                    'Value': str(val), 'Unit': unit}
            data.append(entry)
            smilist.add(smi)

        f = open(OUTNAME, 'w')
        f.write(json.dumps(data,indent=True))
        f.close()

