import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pickle
import sys
import timeit
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
import argparse
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

DATADIR = '../../../../../../data/external/DLKcat/Data'
OOD_INDICES = json.load(open('./catpred_sequence_ood_test_indices.json'))
TEST_INDICES = {'kcat': 20835, 'km': 37056, 'ki': 10736}
VAL_INDICES = {'kcat': 18520, 'km': 32939, 'ki': 9544}

def custom_collate(batch):
    compounds, adjacencies, proteins, interactions = zip(*batch)
    
    # Pad compounds
    compounds_padded = pad_sequence(compounds, batch_first=True, padding_value=0)
    
    # Pad adjacencies
    max_adj_size = max(adj.size(0) for adj in adjacencies)
    adjacencies_padded = torch.stack([F.pad(adj, (0, max_adj_size - adj.size(0), 0, max_adj_size - adj.size(1))) for adj in adjacencies])
    
    # Pad proteins
    proteins_padded = pad_sequence(proteins, batch_first=True, padding_value=0)
    
    # Stack interactions
    interactions_stacked = torch.stack(interactions)
    
    return compounds_padded, adjacencies_padded, proteins_padded, interactions_stacked

class KcatPrediction(nn.Module):
    def __init__(self, n_fingerprint, n_word, dim, layer_gnn, layer_cnn, window, layer_output):
        super(KcatPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2*window+1,
                                              stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim) for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, 1)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.bmm(A, hs)
        return torch.mean(xs, dim=1)

    def attention_cnn(self, x, xs, layer):
        batch_size, seq_len, dim = xs.size()
        xs = xs.unsqueeze(1)  # Add channel dimension
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = xs.squeeze(1)  # Remove channel dimension
        
        h = torch.relu(self.W_attention(x))
        h = h.unsqueeze(1).expand(-1, seq_len, -1)
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(torch.sum(h * hs, dim=2))
        ys = weights.unsqueeze(2) * hs
        return torch.sum(ys, dim=1)

    def forward(self, compounds, adjacencies, proteins):
        fingerprint_vectors = self.embed_fingerprint(compounds)
        compound_vector = self.gnn(fingerprint_vectors, adjacencies, len(self.W_gnn))
        word_vectors = self.embed_word(proteins)
        protein_vector = self.attention_cnn(compound_vector, word_vectors, len(self.W_cnn))
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for layer in self.W_out:
            cat_vector = torch.relu(layer(cat_vector))
        return self.W_interaction(cat_vector)


class EnzymeDataset(Dataset):
    def __init__(self, compounds, adjacencies, proteins, interactions):
        self.compounds = compounds
        self.adjacencies = adjacencies
        self.proteins = proteins
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        return self.compounds[idx], self.adjacencies[idx], self.proteins[idx], self.interactions[idx]

def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]

def load_tensor_float(file_name, dtype):
    return [dtype(d.tolist()) for d in np.load(file_name + '.npy', allow_pickle=True)]

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for compounds, adjacencies, proteins, interactions in tqdm(train_loader):
        compounds, adjacencies, proteins, interactions = compounds.to(device), adjacencies.to(device), proteins.to(device), interactions.to(device)
        optimizer.zero_grad()
        output = model(compounds, adjacencies, proteins)
        loss = F.mse_loss(output, interactions.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, test_loader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for compounds, adjacencies, proteins, interactions in test_loader:
            compounds, adjacencies, proteins = compounds.to(device), adjacencies.to(device), proteins.to(device)
            output = model(compounds, adjacencies, proteins)
            predictions.extend(output.view(-1).tolist())
            actuals.extend(interactions.tolist())
    predictions, actuals = np.array(predictions), np.array(actuals)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    p1mag = np.mean(np.abs(predictions - actuals) < 1)
    return mae, rmse, r2, p1mag

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('parameter', args.parameter)
    print('seed', args.seed)

    import ipdb
    ipdb.set_trace()
    
    # Load data
    dir_input = f'{DATADIR}/{args.parameter}_input/'
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor_float(dir_input + 'regression', torch.FloatTensor)
    
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'sequence_dict.pickle')
    
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)
    print(f'n_fingerprint: {n_fingerprint}')
    print(f'n_word: {n_word}')

    if args.is_validation:
        # Create datasets
        train_dataset = EnzymeDataset(compounds[:TEST_INDICES[args.parameter]], 
                                      adjacencies[:TEST_INDICES[args.parameter]], 
                                      proteins[:TEST_INDICES[args.parameter]], 
                                      interactions[:TEST_INDICES[args.parameter]])
        dev_dataset = EnzymeDataset(compounds[VAL_INDICES[args.parameter]:TEST_INDICES[args.parameter]], 
                                     adjacencies[VAL_INDICES[args.parameter]:TEST_INDICES[args.parameter]], 
                                     proteins[VAL_INDICES[args.parameter]:TEST_INDICES[args.parameter]], 
                                     interactions[VAL_INDICES[args.parameter]:TEST_INDICES[args.parameter]])
        test_dataset = EnzymeDataset(compounds[TEST_INDICES[args.parameter]:], 
                                     adjacencies[TEST_INDICES[args.parameter]:], 
                                     proteins[TEST_INDICES[args.parameter]:], 
                                     interactions[VAL_INDICES[args.parameter]:TEST_INDICES[args.parameter]])
        dev_indices = OOD_INDICES[args.parameter]

        dev_data = {'compounds':[],'adjacencies':[],'proteins':[],'interactions':[]}
        ind = 0
        for c,a,p,i in test_dataset:
            if ind in dev_indices:
                dev_data['compounds'].append(c)
                dev_data['adjacencies'].append(a)
                dev_data['proteins'].append(p)
                dev_data['interactions'].append(i)
            ind+=1
                
        dev_dataset = EnzymeDataset(**dev_data)

    else:
        # Create datasets
        train_dataset = EnzymeDataset(compounds[:TEST_INDICES[args.parameter]], 
                                      adjacencies[:TEST_INDICES[args.parameter]], 
                                      proteins[:TEST_INDICES[args.parameter]], 
                                      interactions[:TEST_INDICES[args.parameter]])
        test_dataset = EnzymeDataset(compounds[TEST_INDICES[args.parameter]:], 
                                     adjacencies[TEST_INDICES[args.parameter]:], 
                                     proteins[TEST_INDICES[args.parameter]:], 
                                     interactions[TEST_INDICES[args.parameter]:])
        ood_indices_ = OOD_INDICES[args.parameter]
        dev_loaders = []
        clusters = []
        for clus in ood_indices_:
            dev_indices = ood_indices_[clus]
            clusters.append(clus)
            dev_data = {'compounds':[],'adjacencies':[],'proteins':[],'interactions':[]}
            ind = 0
            for c,a,p,i in test_dataset:
                if ind in dev_indices:
                    dev_data['compounds'].append(c)
                    dev_data['adjacencies'].append(a)
                    dev_data['proteins'].append(p)
                    dev_data['interactions'].append(i)
                ind+=1
                    
            dev_dataset = EnzymeDataset(**dev_data)
            dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, 
                            num_workers=args.num_workers, collate_fn=custom_collate)
            
            dev_loaders.append(dev_loader)
            
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             num_workers=args.num_workers, collate_fn=custom_collate)

    # Initialize model and optimizer
    model = KcatPrediction(n_fingerprint, n_word, args.dim, args.layer_gnn, args.layer_cnn, args.window, args.layer_output).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    print('Training')
    for epoch in tqdm(range(1, args.iteration + 1)):
        if epoch % args.decay_interval == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay

        loss_train = train(model, train_loader, optimizer, device)
        mae_test, rmse_test, r2_test, p1_test = test(model, test_loader, device)

        print(f'Epoch {epoch}, Loss: {loss_train:.4f}, Test MAE: {mae_test:.4f}, Test R2: {r2_test:.4f}, Test p1mag: {p1_test:.4f}')

        for i, dev_loader in enumerate(dev_loaders):
            mae_dev, rmse_dev, r2_dev, p1_dev = test(model, dev_loaders[i], device)
            print(f'Dev{clusters[i]} MAE: {mae_dev:.4f}, Dev{clusters[i]} R2: {r2_dev:.4f}, Dev{clusters[i]} p1mag: {p1_dev:.4f}')

        # Save model and results if needed

if __name__ == "__main__":
    import os
    print(os.getcwd())
    parser = argparse.ArgumentParser(description='Kcat Prediction Model')
    parser.add_argument('parameter', type=str, help='Parameter type (kcat, km, or ki)')
    parser.add_argument('--is_validation', action='store_true', help='If doing a validation run or not')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--dim', type=int, default=20, help='Embedding dimension')
    parser.add_argument('--layer_gnn', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--window', type=int, default=11, help='Window size for CNN')
    parser.add_argument('--layer_cnn', type=int, default=3, help='Number of CNN layers')
    parser.add_argument('--layer_output', type=int, default=3, help='Number of output layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay')
    parser.add_argument('--decay_interval', type=int, default=10, help='Decay interval')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--iteration', type=int, default=20, help='Number of iterations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
