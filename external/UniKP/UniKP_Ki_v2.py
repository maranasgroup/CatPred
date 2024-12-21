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
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
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
import multiprocessing
ncpu = multiprocessing.cpu_count()

def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab('./external/UniKP/vocab.pkl')
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
    trfm.load_state_dict(torch.load('./external/UniKP/trfm_12_23000.pkl'))
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

def get_ood_indices(train_clusters, test_clusters):
    return [i for i, cluster in enumerate(test_clusters) if cluster not in train_clusters]

def Kcat_predict(feature_train, Label_train, feature_test, Label_test, train_dataset, test_dataset):
    Ifeature = feature_train
    Ifeature_test = feature_test
    Label = Label_train
    Label_test = Label_test

    print('fitting')
    model = ExtraTreesRegressor(n_estimators=1000,verbose=0, n_jobs=ncpu)
    model.fit(Ifeature, Label)
    Pre_label = model.predict(Ifeature_test)

    results_dict = {'heldout': {}, 'ood': {}}

    # Heldout performance
    MAE = mean_absolute_error(Label_test, Pre_label)
    r2 = r2_score(Label_test, Pre_label)
    errors = np.abs(Label_test - Pre_label)
    p1mag = len(errors[errors < 1]) / len(errors)

    rmse = root_mean_squared_error(Label_test, Pre_label)
    r = stats.pearsonr(Label_test, Pre_label)[0]

    results_dict['heldout']['R2'] = r2
    results_dict['heldout']['MAE'] = MAE
    results_dict['heldout']['p1mag'] = p1mag
    results_dict['heldout']['rmse'] = rmse
    results_dict['heldout']['r'] = r

    # OOD performance for different cluster levels
    for N in [99, 80, 60, 40]:
        train_clusters = set(train_dataset[f'sequence_{N}cluster'])
        test_clusters = test_dataset[f'sequence_{N}cluster']

        OOD_INDICES = get_ood_indices(train_clusters, test_clusters)

        Ifeature_test_ood = Ifeature_test[OOD_INDICES]
        Label_test_ood = Label_test[OOD_INDICES]
        Pre_label_ood = model.predict(Ifeature_test_ood)
        
        MAE_ood = mean_absolute_error(Label_test_ood, Pre_label_ood)
        r2_ood = r2_score(Label_test_ood, Pre_label_ood)
        errors = np.abs(Label_test_ood - Pre_label_ood)
        p1mag_ood = len(errors[errors < 1]) / len(errors)
        rmse_ood = root_mean_squared_error(Label_test_ood, Pre_label_ood)
        r_ood = stats.pearsonr(Label_test_ood, Pre_label_ood)[0]
        
        results_dict['ood'][f'cluster_{N}'] = {
            'R2': r2_ood,
            'MAE': MAE_ood,
            'p1mag': p1mag_ood,
            'rmse': rmse_ood,
            'r': r_ood
        }
        
    return results_dict

def Kcat_predict_Ntimes(feature_train, Label_train, feature_test, Label_test, train_dataset, test_dataset, N=10):
    results_dict_all = []
    
    for i in tqdm(range(N)):
        results = Kcat_predict(feature_train, Label_train, feature_test, Label_test, train_dataset, test_dataset)
        results_dict_all.append(results)
        
    return results_dict_all

def calculate_stats(results_dict_all):
    stats_dict = {'heldout': {}, 'ood': {}}
    
    for dataset in ['heldout', 'ood']:
        if dataset == 'heldout':
            for metric in ['R2', 'MAE', 'p1mag', 'rmse', 'r']:
                values = [d[dataset][metric] for d in results_dict_all]
                mean = np.mean(values)
                stderr = stats.sem(values)
                stats_dict[dataset][metric] = {'mean': mean, 'stderr': stderr}
        else:
            for cluster in ['cluster_99', 'cluster_80', 'cluster_60', 'cluster_40']:
                stats_dict['ood'][cluster] = {}
                for metric in ['R2', 'MAE', 'p1mag', 'rmse', 'r']:
                    values = [d[dataset][cluster][metric] for d in results_dict_all]
                    mean = np.mean(values)
                    stderr = stats.sem(values)
                    stats_dict['ood'][cluster][metric] = {'mean': mean, 'stderr': stderr}
    
    return stats_dict

def print_results(stats_dict):
    print("Results Summary:")
    print("================")
    
    print("\nHeldout Dataset:")
    print("-----------------")
    for metric in ['R2', 'MAE', 'p1mag', 'rmse', 'r']:
        mean = stats_dict['heldout'][metric]['mean']
        stderr = stats_dict['heldout'][metric]['stderr']
        print(f"{metric:<5}: {mean:.4f} ± {stderr:.4f}")
    
    print("\nOOD Datasets:")
    print("--------------")
    for cluster in ['cluster_99', 'cluster_80', 'cluster_60', 'cluster_40']:
        print(f"\n{cluster.upper()}:")
        for metric in ['R2', 'MAE', 'p1mag', 'rmse', 'r']:
            mean = stats_dict['ood'][cluster][metric]['mean']
            stderr = stats_dict['ood'][cluster][metric]['stderr']
            print(f"{metric:<5}: {mean:.4f} ± {stderr:.4f}")


if __name__ == '__main__':
    # Dataset Load
    train = pd.read_csv('../data/CatPred-DB/data/ki/ki-random_trainval.csv')[:]
    test = pd.read_csv('../data/CatPred-DB/data/ki/ki-random_test.csv')[:]
    sequence_train = train['sequence']
    sequence_test = test['sequence']
    smiles_train = train['substrate_smiles']
    smiles_test = test['substrate_smiles']
    Label_train = train['log10ki_mean']
    Label_test = test['log10ki_mean']

    if not os.path.exists('../data/external/UniKP/datasets/catpred_km/ki-random_trainval.pkl') or not os.path.exists('./external/UniKP/datasets/catpred_ki/ki-random_test.pkl'):
        smiles_input_train = smiles_to_vec(smiles_train)
        sequence_input_train = Seq_to_vec(sequence_train)
        feature_train = np.concatenate((smiles_input_train, sequence_input_train), axis=1)
        smiles_input_test = smiles_to_vec(smiles_test)
        sequence_input_test = Seq_to_vec(sequence_test)
        feature_test = np.concatenate((smiles_input_test, sequence_input_test), axis=1)
        with open('../data/external/UniKP/datasets/catpred_ki/ki-random_trainval.pkl', "wb") as f:
            pickle.dump(feature_train, f)
        with open('../data/external/UniKP/datasets/catpred_ki/ki-random_test.pkl', "wb") as f:
            pickle.dump(feature_test, f)
    else:
        with open('../data/external/UniKP/datasets/catpred_ki/ki-random_trainval.pkl', "rb") as f:
            feature_train = pickle.load(f)
        with open('../data/external/UniKP/datasets/catpred_ki/ki-random_test.pkl', "rb") as f:
            feature_test = pickle.load(f)

    feature_train = np.array(feature_train)
    Label_train = np.array(Label_train)
    feature_test = np.array(feature_test)
    Label_test = np.array(Label_test)
    results_dict_all = Kcat_predict_Ntimes(feature_train, Label_train, feature_test, Label_test, train, test, N=5)
    # Calculate statistics
    stats_dict = calculate_stats(results_dict_all)

    # Print results
    print_results(stats_dict)