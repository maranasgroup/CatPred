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

OOD_INDICES = [3,
 15,
 28,
 36,
 45,
 48,
 50,
 71,
 82,
 83,
 85,
 86,
 92,
 107,
 109,
 131,
 134,
 135,
 142,
 144,
 146,
 147,
 163,
 173,
 180,
 185,
 187,
 203,
 205,
 211,
 212,
 220,
 228,
 233,
 235,
 236,
 242,
 247,
 257,
 260,
 266,
 276,
 297,
 301,
 302,
 306,
 313,
 314,
 327,
 331,
 343,
 346,
 350,
 356,
 382,
 388,
 395,
 399,
 402,
 403,
 413,
 427,
 434,
 435,
 437,
 438,
 447,
 454,
 466,
 470,
 497,
 503,
 508,
 510,
 511,
 513,
 514,
 518,
 526,
 538,
 541,
 543,
 563,
 564,
 565,
 567,
 568,
 572,
 580,
 590,
 591,
 607,
 610,
 627,
 633,
 636,
 640,
 642,
 656,
 660,
 668,
 690,
 697,
 705,
 706,
 716,
 726,
 728,
 741,
 742,
 746,
 754,
 759,
 769,
 770,
 776,
 778,
 783,
 788,
 789,
 795,
 797,
 800,
 803,
 810,
 815,
 816,
 832,
 840,
 844,
 848,
 849,
 873,
 874,
 877,
 879,
 882,
 891,
 903,
 904,
 908,
 914,
 916,
 926,
 938,
 941,
 949,
 961,
 963,
 970,
 973,
 985,
 990,
 996,
 1019,
 1020,
 1021,
 1022,
 1030,
 1032,
 1039,
 1050,
 1056,
 1072,
 1085,
 1089,
 1090,
 1100,
 1110,
 1112,
 1115,
 1132,
 1138,
 1149,
 1157,
 1199,
 1210,
 1213,
 1240,
 1248,
 1251,
 1271,
 1276,
 1280,
 1290,
 1292,
 1293,
 1309,
 1320,
 1331,
 1341,
 1346,
 1351,
 1353,
 1377,
 1380,
 1382,
 1388,
 1394,
 1395,
 1396,
 1408,
 1409,
 1421,
 1424,
 1430,
 1440,
 1443,
 1447,
 1461,
 1468,
 1469,
 1473,
 1474,
 1477,
 1478,
 1488,
 1494,
 1499,
 1502,
 1503,
 1520,
 1535,
 1538,
 1541,
 1551,
 1553,
 1560,
 1572,
 1573,
 1576,
 1580,
 1592,
 1595,
 1597,
 1610,
 1612,
 1614,
 1652,
 1661,
 1662,
 1671,
 1673,
 1675,
 1685,
 1688,
 1698,
 1704,
 1708,
 1724,
 1726,
 1737,
 1738,
 1739,
 1742,
 1749,
 1751,
 1756,
 1759,
 1760,
 1761,
 1762,
 1764,
 1768,
 1771,
 1777,
 1778,
 1788,
 1801,
 1804,
 1815,
 1817,
 1822,
 1837,
 1846,
 1847,
 1848,
 1849,
 1853,
 1855,
 1879,
 1882,
 1885,
 1893,
1900,
 1908,
 1910,
 1913,
 1924,
 1930,
 1944,
 1965,
 1972,
 1974,
 1989,
 2004,
 2014,
 2016,
 2020,
 2035,
 2040,
 2041,
 2047,
 2062,
 2066,
 2080,
 2082,
 2083,
 2096,
 2097,
 2108,
 2115,
 2120,
 2124,
 2128,
 2131,
 2142,
 2156,
 2166,
 2167,
 2173,
 2174,
 2178,
 2191,
 2202,
 2204,
 2207,
 2219,
 2228,
 2231,
 2240,
 2241,
 2248,
 2252,
 2259,
 2265,
 2273,
 2278,
 2281,
 2301,
 2302,
 2303,
 2304,
 2314]

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
    Ifeature = Ifeature_ini[:20835]
    Ifeature_test = Ifeature_ini[20835:]
    Label = Label_ini[:20835]
    Label_test = Label_ini[20835:]
    
    print('fitting')
    model = ExtraTreesRegressor(n_estimators=1000,verbose=10,n_jobs=30)
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
    
    Ifeature_test_ood = Ifeature_test[OOD_INDICES]
    Label_test_ood = Label_test[OOD_INDICES]
    Pre_label_ood = model.predict(Ifeature_test_ood)
    MAE_ood = mean_absolute_error(Label_test_ood, Pre_label_ood)
    r2_ood = r2_score(Label_test_ood, Pre_label_ood)
    
    errors = np.abs(Label_test_ood-Pre_label_ood)
    p1mag_ood = len(errors[errors<1])/len(errors)
    
    results_dict['ood']['R2'] = r2_ood
    results_dict['ood']['MAE'] = MAE_ood
    results_dict['ood']['p1mag'] = p1mag_ood
    
    # ipdb.set_trace()
    
    print(results_dict)
    
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
    datasets = pd.read_csv('./kcat-random_trainvaltest.csv')
    # print(datasets)
    # ipdb.set_trace()
    sequence = datasets['sequence']
    smiles = datasets['reactant_smiles']
    Label = datasets['log10kcat_max']
    print(len(smiles), len(Label))
    
    if not os.path.exists('./catpred_kcat/kcat-random_trainvaltest_feats.pkl'):
        smiles_input = smiles_to_vec(smiles)
        sequence_input = Seq_to_vec(sequence)
        feature = np.concatenate((smiles_input, sequence_input), axis=1)
        with open("./catpred_kcat/kcat-random_trainvaltest_feats.pkl", "wb") as f:
            pickle.dump(feature, f)
    else:
        with open("./catpred_kcat/kcat-random_trainvaltest_feats.pkl", "rb") as f:
            feature = pickle.load(f)
            
    feature = np.array(feature)
    Label = np.array(Label)
    
    results_dict_all = Kcat_predict_Ntimes(feature, Label, N=10) # To account for stderr
    
    # Calculate statistics
    stats_dict = calculate_stats(results_dict_all)

    # Print results
    print_results(stats_dict)
