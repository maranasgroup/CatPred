import pandas as pd
import argparse
from data_utils import featurize
from skops.io import load
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input csv file", type=str, required=True)
    parser.add_argument("-par", "--parameter", help="parameter to predict", 
                        type=str, required=True)

    args, unparsed = parser.parse_known_args()
    parser = argparse.ArgumentParser()

    return args

args = parse_args()

print(args)
root_path = '.'
data_dir = './data/'

dfin = pd.read_csv(args.input)

df = dfin.copy()
X, df = featurize(df, args.parameter.upper())
df.to_pickle(f'{args.input[:-4]}_feats.pkl')

model = load(f'./models/{args.parameter.upper()}_rf_500_01.skops')

print('Making predictions ...')
preds = model.predict(X)

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