"""Loads a trained chemprop model checkpoint and makes predictions on a dataset."""

from chemprop.train import chemprop_predict_and_fp
import ipdb

if __name__ == '__main__':
    results = chemprop_predict_and_fp()
    # ipdb.set_trace()
    
