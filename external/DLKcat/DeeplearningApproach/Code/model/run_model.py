#!/usr/bin/python
# coding: utf-8

# Author: LE YUAN
# Date: 2020-10-23

import pickle
import sys
import timeit
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error,r2_score
from tqdm import tqdm

OOD_INDICES_KCAT = [3,
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

class KcatPrediction(nn.Module):
    def __init__(self):
        super(KcatPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        # self.W_interaction = nn.Linear(2*dim, 2)
        self.W_interaction = nn.Linear(2*dim, 1)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)
        # print(interaction)

        return interaction

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)
        # print(predicted_interaction)

        if train:
            loss = F.mse_loss(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_values = correct_interaction.to('cpu').data.numpy()
            predicted_values = predicted_interaction.to('cpu').data.numpy()[0]
            # correct_values = np.concatenate(correct_values)
            # predicted_values = np.concatenate(predicted_values)
            # ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            # predicted_values = list(map(lambda x: np.argmax(x), ys))
            # print(correct_values)
            # print(predicted_values)
            # predicted_scores = list(map(lambda x: x[1], ys))
            return correct_values, predicted_values


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for data in tqdm(dataset):
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        SAE = 0  # sum absolute error.
        testY, testPredict = [], []
        for data in dataset :
            (correct_values, predicted_values) = self.model(data, train=False)
            correct_values = math.log10(math.pow(2,correct_values))
            predicted_values = math.log10(math.pow(2,predicted_values))
            SAE += np.abs(predicted_values-correct_values)
            # SAE += sum(np.abs(predicted_values-correct_values))
            testY.append(correct_values)
            testPredict.append(predicted_values)
        MAE = SAE / N  # mean absolute error.
        rmse = np.sqrt(mean_squared_error(testY,testPredict))
        r2 = r2_score(testY,testPredict)
        errors = np.abs(np.array(testY)-np.array(testPredict))
        p1mag = len(errors[errors<1])/len(errors)
        return MAE, rmse, r2, p1mag

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, MAEs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = sys.argv[1:]
    (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval,
     iteration) = map(int, [dim, layer_gnn, window, layer_cnn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    # print(type(radius))

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('../../Data/input/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'regression', torch.FloatTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'sequence_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)
    print(n_fingerprint)  # 3958
    print(n_word)  # 8542
    # 394 and 474 when radius=1 and ngram=2

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    # dataset = shuffle_dataset(dataset, 1234)
    # dataset_train, dataset_ = split_dataset(dataset, 0.8)
    # dataset_dev, dataset_test = split_dataset(dataset_, 0.5)
    dataset_train = dataset[:20835]
    dataset_test = dataset[20835:]
    dataset_dev = [dataset_test[each] for each in OOD_INDICES_KCAT] #ood test

    """Set a model."""
    torch.manual_seed(1234)
    model = KcatPrediction().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    file_MAEs = '../../Data/Results/output/MAEs--' + setting + '.txt'
    file_model = '../../Data/Results/output/' + setting
    # MAEs = ('Epoch\tTime(sec)\tLoss_train\tMAE_dev\t'
    #         'MAE_test\tPrecision_test\tRecall_test')
    MAEs = ('Epoch\tTime(sec)\tLoss_train\tMAE_dev\tMAE_test\tRMSE_dev\tRMSE_test\tR2_dev\tR2_test\tp1_dev\tp1_test')
    with open(file_MAEs, 'w') as f:
        f.write(MAEs + '\n')

    """Start training."""
    print('Training...')
    print(MAEs)
    start = timeit.default_timer()

    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)
        MAE_dev, RMSE_dev, R2_dev, p1dev = tester.test(dataset_dev)
        MAE_test, RMSE_test, R2_test, p1test = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        MAEs = [epoch, time, loss_train, MAE_dev,
                MAE_test, RMSE_dev, RMSE_test, R2_dev, R2_test, p1dev, p1test]
        tester.save_MAEs(MAEs, file_MAEs)
        tester.save_model(model, file_model)

        print('\t'.join(map(str, MAEs)))
#!/usr/bin/python
# coding: utf-8

# Author: LE YUAN
# Date: 2020-10-23

import pickle
import sys
import timeit
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error,r2_score
from tqdm import tqdm

OOD_INDICES_KCAT = [3,
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

class KcatPrediction(nn.Module):
    def __init__(self):
        super(KcatPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        # self.W_interaction = nn.Linear(2*dim, 2)
        self.W_interaction = nn.Linear(2*dim, 1)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)
        # print(interaction)

        return interaction

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)
        # print(predicted_interaction)

        if train:
            loss = F.mse_loss(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_values = correct_interaction.to('cpu').data.numpy()
            predicted_values = predicted_interaction.to('cpu').data.numpy()[0]
            # correct_values = np.concatenate(correct_values)
            # predicted_values = np.concatenate(predicted_values)
            # ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            # predicted_values = list(map(lambda x: np.argmax(x), ys))
            # print(correct_values)
            # print(predicted_values)
            # predicted_scores = list(map(lambda x: x[1], ys))
            return correct_values, predicted_values


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for data in tqdm(dataset):
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        SAE = 0  # sum absolute error.
        testY, testPredict = [], []
        for data in dataset :
            (correct_values, predicted_values) = self.model(data, train=False)
            correct_values = math.log10(math.pow(2,correct_values))
            predicted_values = math.log10(math.pow(2,predicted_values))
            SAE += np.abs(predicted_values-correct_values)
            # SAE += sum(np.abs(predicted_values-correct_values))
            testY.append(correct_values)
            testPredict.append(predicted_values)
        MAE = SAE / N  # mean absolute error.
        rmse = np.sqrt(mean_squared_error(testY,testPredict))
        r2 = r2_score(testY,testPredict)
        errors = np.abs(np.array(testY)-np.array(testPredict))
        p1mag = len(errors[errors<1])/len(errors)
        return MAE, rmse, r2, p1mag

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, MAEs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = sys.argv[1:]
    (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval,
     iteration) = map(int, [dim, layer_gnn, window, layer_cnn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    # print(type(radius))

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('../../Data/input/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'regression', torch.FloatTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'sequence_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)
    print(n_fingerprint)  # 3958
    print(n_word)  # 8542
    # 394 and 474 when radius=1 and ngram=2

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    # dataset = shuffle_dataset(dataset, 1234)
    # dataset_train, dataset_ = split_dataset(dataset, 0.8)
    # dataset_dev, dataset_test = split_dataset(dataset_, 0.5)
    dataset_train = dataset[:20835]
    dataset_test = dataset[20835:]
    dataset_dev = [dataset_test[each] for each in OOD_INDICES_KCAT] #ood test

    """Set a model."""
    torch.manual_seed(1234)
    model = KcatPrediction().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    file_MAEs = '../../Data/Results/output/MAEs--' + setting + '.txt'
    file_model = '../../Data/Results/output/' + setting
    # MAEs = ('Epoch\tTime(sec)\tLoss_train\tMAE_dev\t'
    #         'MAE_test\tPrecision_test\tRecall_test')
    MAEs = ('Epoch\tTime(sec)\tLoss_train\tMAE_dev\tMAE_test\tRMSE_dev\tRMSE_test\tR2_dev\tR2_test\tp1_dev\tp1_test')
    with open(file_MAEs, 'w') as f:
        f.write(MAEs + '\n')

    """Start training."""
    print('Training...')
    print(MAEs)
    start = timeit.default_timer()

    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)
        MAE_dev, RMSE_dev, R2_dev, p1dev = tester.test(dataset_dev)
        MAE_test, RMSE_test, R2_test, p1test = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        MAEs = [epoch, time, loss_train, MAE_dev,
                MAE_test, RMSE_dev, RMSE_test, R2_dev, R2_test, p1dev, p1test]
        tester.save_MAEs(MAEs, file_MAEs)
        tester.save_model(model, file_model)

        print('\t'.join(map(str, MAEs)))
