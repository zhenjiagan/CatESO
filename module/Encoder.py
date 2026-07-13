import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from dgl.nn import GraphConv

class ResGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, activation=F.relu):
        super(ResGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activation = activation
        
        self.layers.append(GraphConv(in_feats, hidden_feats[0], allow_zero_in_degree=True))
        self.norms.append(nn.LayerNorm(hidden_feats[0]))
        
        for i in range(len(hidden_feats) - 1):
            self.layers.append(GraphConv(hidden_feats[i], hidden_feats[i+1], allow_zero_in_degree=True))
            self.norms.append(nn.LayerNorm(hidden_feats[i+1]))

    def forward(self, g, features):
        h = features
        
        for i, layer in enumerate(self.layers):
            h_in = h 
            
            h = layer(g, h)        
            h = self.norms[i](h)   
            h = self.activation(h) 
            
            if i > 0 and h.shape == h_in.shape:
                h = h + h_in
                
        return h


class Encoder_drug(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(Encoder_drug, self).__init__()
        
        if hidden_feats is None:
            hidden_feats = [128, 128, 128]
        
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        
        self.gnn = ResGCN(
            in_feats=dim_embedding, 
            hidden_feats=hidden_feats, 
            activation=activation if activation else F.relu
        )
        
        self.output_feats = hidden_feats[-1]
    
    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        
        node_feats = self.init_transform(node_feats)
        
        node_feats = self.gnn(batch_graph, node_feats)
        
        node_counts = batch_graph.batch_num_nodes()
        node_splits = torch.split(node_feats, node_counts.tolist())
        node_feats = pad_sequence(node_splits, batch_first=True, padding_value=0.0)

        return node_feats

class Encoder_protein(nn.Module):
    def __init__(self, embedding_dim=320, hidden_dim=320, target_dim=128):
        super(Encoder_protein, self).__init__()
        self.output_layer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, x):
        x = self.output_layer(x)
        return x

