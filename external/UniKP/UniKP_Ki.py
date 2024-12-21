import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
import json
from transformers import T5EncoderModel, T5Tokenizer
import re
import os
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import ipdb
from tqdm import tqdm
from scipy import stats

TEST_INDS_OOD = [1,
 2,
 10,
 25,
 31,
 49,
 51,
 61,
 66,
 103,
 108,
 126,
 131,
 152,
 157,
 159,
 188,
 192,
 196,
 198,
 200,
 201,
 203,
 213,
 228,
 230,
 236,
 241,
 248,
 255,
 258,
 267,
 290,
 292,
 295,
 298,
 304,
 307,
 309,
 315,
 325,
 326,
 333,
 344,
 345,
 371,
 376,
 382,
 397,
 398,
 413,
 417,
 418,
 431,
 433,
 435,
 443,
 454,
 464,
 475,
 498,
 518,
 519,
 521,
 524,
 535,
 539,
 540,
 548,
 554,
 567,
 573,
 576,
 590,
 616,
 623,
 624,
 629,
 644,
 656,
 676,
 687,
 696,
 701,
 706,
 740,
 748,
 750,
 751,
 752,
 766,
 780,
 782,
 786,
 790,
 807,
 809,
 810,
 812,
 818,
 822,
 830,
 835,
 846,
 863,
 866,
 874,
 876,
 919,
 920,
 930,
 934,
 960,
 964,
 968,
 969,
 973,
 1004,
 1007,
 1021,
 1026,
 1030,
 1043,
 1047,
 1051,
 1053,
 1069,
 1077,
 1078,
 1086,
 1088,
 1095,
 1099,
 1109,
 1111,
 1119,
 1124,
 1127,
 1173,
 1175]

def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('vocab.pkl')
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm)>218:
            print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109]+sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1]*len(ids)
        padding = [pad_index]*(seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg
    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a,b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load('trfm_12_23000.pkl'))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X


def Seq_to_vec(Sequence):
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    # 'cuda:0' if torch.cuda.is_available() else
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in tqdm(range(len(sequences_Example))):
        # print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in tqdm(range(len(features))):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    return features_normalize


def Kcat_predict(Ifeature_ini, Label_ini):
    # Myseed = random.randint(0, 1000)
    # print(Myseed)
    # Ifeature, Ifeature_test, Label, Label_test = train_test_split(Ifeature_ini, Label_ini, test_size=0.2,
    #                                                               random_state=Myseed)
    Ifeature = Ifeature_ini[:10736]
    Ifeature_test = Ifeature_ini[10736:]
    Label = Label_ini[:10736]
    Label_test = Label_ini[10736:]
    
    print('fitting')
    model = ExtraTreesRegressor()
    model.fit(Ifeature, Label)
    Pre_label = model.predict(Ifeature_test)
    MAE = mean_absolute_error(Label_test, Pre_label)
    r2 = r2_score(Label_test, Pre_label)
    errors = np.abs(Label_test-Pre_label)
    p1mag = len(errors[errors<1])/len(errors)
    
    results_dict = {'heldout': {}, 'ood': {}}
    results_dict['heldout']['R2'] = r2
    results_dict['heldout']['MAE'] = MAE
    results_dict['heldout']['p1mag'] = p1mag
    
    Ifeature_test_ood = Ifeature_test[TEST_INDS_OOD]
    Label_test_ood = Label_test[TEST_INDS_OOD]
    Pre_label = model.predict(Ifeature_test_ood)
    MAE = mean_absolute_error(Label_test_ood, Pre_label)
    r2 = r2_score(Label_test_ood, Pre_label)
    
    errors = np.abs(Label_test_ood-Pre_label)
    p1mag = len(errors[errors<1])/len(errors)
    
    results_dict['ood']['R2'] = r2
    results_dict['ood']['MAE'] = MAE
    results_dict['ood']['p1mag'] = p1mag
    
    return results_dict

def Kcat_predict_Ntimes(feature, Label, N):
    results_dict_all = []
    
    for i in range(N):
        results = Kcat_predict(feature, Label)
        results_dict_all.append(results)
        
    return results_dict_all

def calculate_stats(results_dict_all):
    stats_dict = {'heldout': {}, 'ood': {}}
    
    for dataset in ['heldout', 'ood']:
        for metric in ['R2', 'MAE', 'p1mag']:
            values = [d[dataset][metric] for d in results_dict_all]
            mean = np.mean(values)
            stderr = stats.sem(values)
            stats_dict[dataset][metric] = {'mean': mean, 'stderr': stderr}
    
    return stats_dict

def print_results(stats_dict):
    print("Results Summary:")
    print("================")
    
    for dataset in ['heldout', 'ood']:
        print(f"\n{dataset.upper()} Dataset:")
        print("-----------------")
        for metric in ['R2', 'MAE', 'p1mag']:
            mean = stats_dict[dataset][metric]['mean']
            stderr = stats_dict[dataset][metric]['stderr']
            print(f"{metric:<5}: {mean:.4f} ± {stderr:.4f}")

if __name__ == '__main__':
    # Dataset Load
    datasets = pd.read_csv('./ki-random_trainvaltest.csv')
    # print(datasets)
    # ipdb.set_trace()
    sequence = datasets['sequence']
    smiles = datasets['substrate_smiles']
    Label = datasets['log10ki_mean']
    print(len(smiles), len(Label))
    
    if not os.path.exists('./catpred_ki/ki-random_trainvaltest_feats.pkl'):
        smiles_input = smiles_to_vec(smiles)
        sequence_input = Seq_to_vec(sequence)
        feature = np.concatenate((smiles_input, sequence_input), axis=1)
        with open("./catpred_ki/ki-random_trainvaltest_feats.pkl", "wb") as f:
            pickle.dump(feature, f)
    else:
        with open("./catpred_ki/ki-random_trainvaltest_feats.pkl", "rb") as f:
            feature = pickle.load(f)
            
    feature = np.array(feature)
    Label = np.array(Label)
    
    results_dict_all = Kcat_predict_Ntimes(feature, Label, N=10) # To account for stderr
    
    # Calculate statistics
    stats_dict = calculate_stats(results_dict_all)

    # Print results
    print_results(stats_dict)
