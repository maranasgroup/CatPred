import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "--data_dir", help="data directory", type=str, required=True)
    parser.add_argument("--par", "--par", help="parameter to predict", 
                        type=str, required=True)

    args, unparsed = parser.parse_known_args()
    parser = argparse.ArgumentParser()

    return args

args = parse_args()

param = args.par.upper()
data_dir = args.data_dir

df = pd.read_csv(f'{data_dir}/catpred-db_{param}.csv')

df_train, df_test = train_test_split(df,
                                     test_size=0.10,
                                     random_state=0)
df_train.reset_index(drop=True,inplace=True)
df_test.reset_index(drop=True,inplace=True)

df_train.to_csv(f'{data_dir}/catpred-db_{param}_train.csv')
df_test.to_csv(f'{data_dir}/catpred-db_{param}_test.csv')

