import pandas as pd
import argparse
import json
def parse_args():
    """Prepare argument parser.

    Args:

    Return:

    """
    parser = argparse.ArgumentParser(
        description="Generate json records for test file with only sequences"
    )
    parser.add_argument(
        "--data_file",
        help="Path to csv file",
        required=True,
    )
    
    parser.add_argument("--out_file", help="output file for json records")

    args = parser.parse_args()
    return args

args = parse_args()
df = pd.read_csv(args.data_file)
assert('pdbpath' in df.columns)
assert('sequence' in df.columns)

import json
dic_full = {}
for ind, row in df.iterrows():
    dic = {}
    dic['name'] = row.pdbpath
    dic['seq'] = row.sequence
    dic_full[row.pdbpath] = dic

f = open(args.out_file,'w')
f.write(json.dumps(dic_full))
f.close()
