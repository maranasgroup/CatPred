import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
import json
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import random
import pickle
import math
from sklearn.model_selection import KFold


def Kcat_predict(Ifeature, Label):
    for i in range(5):
        model = ExtraTreesRegressor()
        model.fit(Ifeature, Label)
        with open('PreKcat_new/'+str(i)+"_model.pkl", "wb") as f:
            pickle.dump(model, f)


if __name__ == '__main__':
    with open('Kcat_combination_0918_wildtype_mutant.json', 'r') as file:
        datasets = json.load(file)
    # print(len(datasets))
    Label = [float(data['Value']) for data in datasets]
    Smiles = [data['Smiles']for data in datasets]
    for i in range(len(Label)):
        if Label[i] == 0:
            Label[i] = -10000000000
        else:
            Label[i] = math.log(Label[i], 10)
    with open("PreKcat_new/features_16838_PreKcat.pkl", "rb") as f:
        feature = pickle.load(f)
    Label = np.array(Label)
    Label_new = []
    feature_new = []
    for i in range(len(Label)):
        if -10000000000 < Label[i] and '.' not in Smiles[i]:
            Label_new.append(Label[i])
            feature_new.append(feature[i])
    print(len(Label_new))
    Label_new = np.array(Label_new)
    feature_new = np.array(feature_new)
    Kcat_predict(feature_new, Label_new)
