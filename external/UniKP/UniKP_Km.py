import torch
import os
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import ipdb
from tqdm import tqdm
import numpy as np
from scipy import stats

TEST_INDS_OOD = [8,
 12,
 15,
 22,
 34,
 39,
 52,
 69,
 72,
 83,
 87,
 125,
 134,
 138,
 149,
 153,
 157,
 168,
 173,
 175,
 185,
 198,
 212,
 222,
 224,
 242,
 254,
 256,
 267,
 287,
 296,
 304,
 305,
 310,
 320,
 325,
 327,
 345,
 350,
 366,
 378,
 387,
 389,
 393,
 400,
 414,
 421,
 423,
 427,
 437,
 438,
 445,
 446,
 465,
 490,
 496,
 510,
 514,
 519,
 521,
 534,
 539,
 543,
 563,
 598,
 612,
 614,
 615,
 616,
 618,
 622,
 628,
 633,
 635,
 638,
 645,
 647,
 649,
 650,
 651,
 659,
 673,
 680,
 684,
 685,
 687,
 689,
 692,
 698,
 707,
 720,
 729,
 732,
 740,
 742,
 751,
 755,
 774,
 785,
 789,
 792,
 797,
 805,
 810,
 813,
 842,
 847,
 859,
 860,
 871,
 888,
 889,
 895,
 899,
 900,
 902,
 907,
 914,
 915,
 918,
 933,
 955,
 965,
 970,
 982,
 992,
 998,
 1004,
 1029,
 1041,
 1047,
 1067,
 1071,
 1080,
 1081,
 1089,
 1091,
 1093,
 1105,
 1114,
 1123,
 1137,
 1156,
 1158,
 1159,
 1164,
 1168,
 1177,
 1188,
 1203,
 1237,
 1239,
 1247,
 1250,
 1258,
 1268,
 1269,
 1270,
 1272,
 1282,
 1287,
 1292,
 1303,
 1320,
 1324,
 1338,
 1354,
 1368,
 1381,
 1384,
 1385,
 1400,
 1406,
 1415,
 1416,
 1420,
 1421,
 1428,
 1429,
 1430,
 1458,
 1469,
 1504,
 1508,
 1534,
 1537,
 1543,
 1553,
 1557,
 1566,
 1569,
 1571,
 1586,
 1596,
 1603,
 1611,
 1647,
 1656,
 1657,
 1659,
 1672,
 1692,
 1696,
 1699,
 1713,
 1721,
 1732,
 1736,
 1738,
 1754,
 1756,
 1758,
 1764,
 1773,
 1783,
 1784,
 1802,
 1811,
 1815,
 1854,
 1857,
 1858,
 1868,
 1876,
 1882,
 1893,
 1908,
 1921,
 1934,
 1935,
 1941,
 1948,
 1953,
 1955,
 1959,
 1968,
 1970,
 1979,
 1986,
 1995,
 2001,
 2004,
 2008,
 2011,
 2024,
 2031,
 2033,
 2036,
 2053,
 2058,
 2059,
 2091,
 2107,
 2109,
 2110,
 2124,
 2127,
 2130,
 2132,
 2151,
 2160,
 2163,
 2166,
 2173,
 2175,
 2178,
 2182,
 2209,
 2211,
 2220,
 2230,
 2255,
 2267,
 2277,
 2280,
 2287,
 2300,
 2311,
 2314,
 2322,
 2331,
 2343,
 2345,
 2373,
 2378,
 2386,
 2412,
 2414,
 2430,
 2447,
 2448,
 2449,
 2450,
 2486,
 2487,
 2508,
 2511,
 2514,
 2516,
 2519,
 2540,
 2544,
 2550,
 2569,
 2581,
 2594,
 2611,
 2616,
 2625,
 2629,
 2639,
 2656,
 2662,
 2677,
 2680,
 2683,
 2684,
 2691,
 2692,
 2702,
 2706,
 2709,
 2716,
 2721,
 2727,
 2730,
 2740,
 2763,
 2769,
 2776,
 2780,
 2795,
 2833,
 2834,
 2846,
 2847,
 2851,
 2854,
 2862,
 2868,
 2873,
 2875,
 2885,
 2913,
 2919,
 2924,
 2933,
 2941,
 2952,
 2957,
 2977,
 2980,
 2981,
 2983,
 2984,
 2996,
 3015,
 3017,
 3020,
 3034,
 3036,
 3047,
 3054,
 3059,
 3070,
 3091,
 3093,
 3099,
 3103,
 3120,
 3127,
 3144,
 3148,
 3169,
 3191,
 3197,
 3216,
 3227,
 3257,
 3268,
 3271,
 3280,
 3281,
 3282,
 3297,
 3300,
 3304,
 3314,
 3324,
 3354,
 3355,
 3359,
 3378,
 3421,
 3436,
 3439,
 3440,
 3444,
 3449,
 3451,
 3458,
 3461,
 3474,
 3512,
 3515,
 3518,
 3523,
 3538,
 3543,
 3552,
 3555,
 3557,
 3562,
 3565,
 3577,
 3583,
 3594,
 3601,
 3602,
 3646,
 3656,
 3657,
 3671,
 3674,
 3678,
 3682,
 3694,
 3699,
3707,
 3710,
 3730,
 3750,
 3760,
 3764,
 3787,
 3789,
 3796,
 3805,
 3812,
 3814,
 3828,
 3829,
 3836,
 3839,
 3841,
 3845,
 3847,
 3856,
 3880,
 3889,
 3899,
 3902,
 3915,
 3928,
 3936,
 3937,
 3939,
 3950,
 3954,
 3956,
 3964,
 3973,
 3977,
 3980,
 4011,
 4015,
 4026,
 4032,
 4035,
 4036,
 4045,
 4048,
 4051,
 4055,
 4074,
 4088,
 4096,
 4100,
 4108]

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
    Ifeature = Ifeature_ini[:37056]
    Ifeature_test = Ifeature_ini[37056:]
    Label = Label_ini[:37056]
    Label_test = Label_ini[37056:]
    
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
    datasets = pd.read_csv('./km-random_trainvaltest.csv')
    sequence = datasets['sequence']
    smiles = datasets['substrate_smiles']
    Label = datasets['log10km_mean']
    print(len(smiles), len(Label))
    
    if not os.path.exists('./catpred_km/km-random_trainvaltest_feats.pkl'):
        smiles_input = smiles_to_vec(smiles)
        sequence_input = Seq_to_vec(sequence)
        feature = np.concatenate((smiles_input, sequence_input), axis=1)
        with open("./catpred_km/km-random_trainvaltest_feats.pkl", "wb") as f:
            pickle.dump(feature, f)
    else:
        with open("./catpred_km/km-random_trainvaltest_feats.pkl", "rb") as f:
            feature = pickle.load(f)
            
    feature = np.array(feature)
    Label = np.array(Label)
    
    results_dict_all = Kcat_predict_Ntimes(feature, Label, N=10) # To account for stderr
    
    # Calculate statistics
    stats_dict = calculate_stats(results_dict_all)

    # Print results
    print_results(stats_dict)

