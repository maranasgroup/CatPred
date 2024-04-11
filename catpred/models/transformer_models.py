import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
import ipdb

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, dim_hidden, dim_inner_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.layer_1 = nn.Conv1d(dim_hidden, dim_inner_hidden, 1)  # position-wise
        self.layer_2 = nn.Conv1d(dim_inner_hidden, dim_hidden, 1)  # position-wise
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_hidden)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        # print("Input of fnn {}".format(x.size()))
        # print("transposed Input of fnn {}".format(x.transpose(1, 2).size()))
        output = self.relu(self.layer_1(x.transpose(1, 2)))
        # print("First convolution of fnn {}".format(output.size()))
        output = self.layer_2(output).transpose(2, 1)
        # print("Second convolution of fnn {}".format(output.size()))
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    ''' Transformer encoder layer '''

    def __init__(self, dim_model, dim_inner_hidden, qty_head, dropout = 0.1, attn_dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(dim_model, qty_head, 
                                                    dropout=attn_dropout, batch_first=True)
        self.feedforward = PositionwiseFeedForward(dim_model, dim_inner_hidden, dropout)

    def forward(self, q, k, v):
        output, attention = self.self_attention(q, k, v)
        output = self.feedforward(output)
        return output, attention

class TransformerEncoder(nn.Module):
    ''' A neural network Transformer Encoder '''

    def __init__(self, vocab_size, qty_encoder_layer=3, qty_attention_head=8,
                 dim_vocab_embedding=256, dim_model=256, dim_inner_hidden=128,
                 dropout=0.2, attn_dropout=0.1, embedding=False):
        super(TransformerEncoder, self).__init__()

        self.dim_model = dim_model

        # Embedding containing sentence order information
        self.position_encoder = RotaryEmbedding(dim=32)

        # Embedding vector of words. TODO: test with word2vec
        self.embedding_layer = nn.Embedding(vocab_size, dim_vocab_embedding, padding_idx=20)

        # Create a set of encoder layers, given the quantity informed in
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dim_model, dim_inner_hidden, 
                         qty_attention_head, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(qty_encoder_layer)
        ])

        # whether do embedding before attention module
        self.embedding = embedding
        print('''Transformer Model:
                    - dim = {}
                    - encoder layers = {}
                    - attention heads = {}
                    '''.format(dim_vocab_embedding, qty_encoder_layer, qty_attention_head))

    def _get_qkv(self, input_tensor):
        q = self.position_encoder.rotate_queries_or_keys(input_tensor,seq_dim=1)
        k = self.position_encoder.rotate_queries_or_keys(input_tensor,seq_dim=1)
        return q, k, input_tensor

    def get_trainable_parameters(self):
        """ Avoid updating the position encoding """
        position_parameters = set(map(id, self.position_encoder.parameters()))
        return (p for p in self.parameters() if id(p) not in position_parameters)

    def forward(self, sequence):
        if(self.embedding):
            # lookup word embedding layer
            word_embedding = self.embedding_layer(sequence)
        else:
            word_embedding = sequence

        # ipdb.set_trace()
        q,k,v = self._get_qkv(word_embedding)

        for encoder_layer in self.encoder_layers:
            encoder_output, attentions = encoder_layer(q,k,v)
            q,k,v = self._get_qkv(encoder_output)
            
        return encoder_output