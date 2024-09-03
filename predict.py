"""Loads a trained catpred model checkpoint and makes predictions on a dataset."""

from catpred.train import catpred_predict_and_fp
import ipdb

if __name__ == '__main__':
    results = catpred_predict_and_fp()
    # ipdb.set_trace()
    
