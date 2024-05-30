"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, HypergraphConv, GCNConv, global_add_pool, global_max_pool, global_mean_pool
import manifolds


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """
    def __init__(self, manifold, in_features, in_structure, out_features, c_in, num_layers, args, is_hypergraph):
        super(HyperbolicGraphConvolution, self).__init__()

        self.agg = HypAgg(manifold, c_in, in_features, in_structure, out_features, num_layers, args, is_hypergraph)#.to(args.device) 
    
    def forward(self, input):   
        data, is_hypergraph = input
        hyperbolic_g, hyperbolic_g_f, hyperbolic_n_f, hyperbolic_g_s, hyperbolic_n_s = self.agg.forward(data, is_hypergraph)

        return hyperbolic_g, hyperbolic_g_f, hyperbolic_n_f, hyperbolic_g_s, hyperbolic_n_s
    


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """
    def __init__(self, manifold, c, in_features, in_structure, out_features, num_layers, args, is_hypergraph):
      
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
       
        self.embedding_dim = out_features
        if args.readout == 'concat':
            self.embedding_dim *= num_layers
        self.proj_head = nn.Sequential(Linear(self.embedding_dim, self.embedding_dim), ReLU(inplace=True),
                                       Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_g = nn.Sequential(nn.Linear(self.embedding_dim * 2, self.embedding_dim), nn.ReLU(inplace=True),
                                            nn.Linear(self.embedding_dim, self.embedding_dim))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if is_hypergraph:
            self.stackGCNs_f = getattr(Encoder_HyperGNN(in_features, out_features, num_layers, args.pooling, args.readout).to(device), 'forward')
            self.stackGCNs_s = getattr(Encoder_HyperGNN(in_structure, out_features, num_layers, args.pooling, args.readout).to(device), 'forward')
        else:
            if args.GNN_Encoder == 'GCN':
                self.stackGCNs_f = getattr(Encoder_GCN(in_features, out_features, num_layers, args.pooling, args.readout).to(device), 'forward') 
                self.stackGCNs_s = getattr(Encoder_GCN(in_structure, out_features, num_layers, args.pooling, args.readout).to(device), 'forward') 
            else:
                self.stackGCNs_f = getattr(Encoder_GIN(in_features, out_features, num_layers, args.pooling, args.readout).to(device), 'forward') 
                self.stackGCNs_s = getattr(Encoder_GIN(in_structure, out_features, num_layers, args.pooling, args.readout).to(device), 'forward') 

        self.init_emb()
         
    def init_emb(self):              
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                m.weight = manifolds.ManifoldParameter(m.weight, True, self.manifold, self.c)
                if m.bias is not None:
                    m.bias = None

    
    def forward(self, data, is_hypergraph):

        x_tangent_f = self.manifold.logmap0(data.x, c=self.c)  
        x_tangent_s = self.manifold.logmap0(data.x_s, c=self.c)  
     
        if is_hypergraph:
            g_tan_f, n_tan_f = self.stackGCNs_f(x_tangent_f, data.hyperedge_index, data.batch)    
            g_tan_s, n_tan_s = self.stackGCNs_s(x_tangent_s, data.hyperedge_index, data.batch)
        else: 
            g_tan_f, n_tan_f = self.stackGCNs_f(x_tangent_f, data.edge_index, data.batch)    
            g_tan_s, n_tan_s = self.stackGCNs_s(x_tangent_s, data.edge_index, data.batch)

        g_proj_tan = self.proj_head_g(torch.cat((g_tan_f, g_tan_s), 1))
        g_proj_tan_f = self.proj_head(g_tan_f)
        g_proj_tan_s = self.proj_head(g_tan_s)
        n_proj_tan_f = self.proj_head(n_tan_f)
        n_proj_tan_s = self.proj_head(n_tan_s)


        hyperbolic_g_f = self.manifold.proj(self.manifold.expmap0(g_proj_tan_f, c=self.c), c=self.c)
        hyperbolic_g_s = self.manifold.proj(self.manifold.expmap0(g_proj_tan_s, c=self.c), c=self.c)
        hyperbolic_n_f = self.manifold.proj(self.manifold.expmap0(n_proj_tan_f, c=self.c), c=self.c)
        hyperbolic_n_s = self.manifold.proj(self.manifold.expmap0(n_proj_tan_s, c=self.c), c=self.c)
        hyperbolic_g = self.manifold.proj(self.manifold.expmap0(g_proj_tan, c=self.c), c=self.c)

        return hyperbolic_g, hyperbolic_g_f, hyperbolic_n_f, hyperbolic_g_s, hyperbolic_n_s
    
    
    def extra_repr(self):
        return 'c={}'.format(self.c)


class Encoder_GIN(torch.nn.Module):  

    def __init__(self, num_features, dim, num_gc_layers, pooling, readout):
        super(Encoder_GIN, self).__init__()
        self.dim = dim
        self.num_gc_layers = num_gc_layers
        self.pooling = pooling
        self.readout = readout
        self.pool = self.get_pool()

        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)

            self.convs.append(conv)

    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        if self.readout == 'last':
            graph_emb = self.pool(xs[-1], batch)
        elif self.readout == 'concat':
            graph_emb = torch.cat([self.pool(x, batch) for x in xs], 1)
        elif self.readout == 'add':
            graph_emb = 0
            for x in xs:
                graph_emb += self.pool(x, batch)
        
        return graph_emb, torch.cat(xs, 1)   
    
    def get_pool(self):
        if self.pooling == 'add':
            pool = global_add_pool
        elif self.pooling == 'max':
            pool = global_max_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(self.pooling))
        return pool


class Encoder_HyperGNN(torch.nn.Module):   
    def __init__(self, input_dim, hidden_dim, num_gc_layers, pooling, readout):
        super(Encoder_HyperGNN, self).__init__()

        self.num_node_features = input_dim

        self.nhid = hidden_dim
        self.enhid = hidden_dim                                

        self.num_convs = num_gc_layers
        self.pooling = pooling
        self.readout = readout

        self.convs = self.get_convs()
        self.pool = self.get_pool()


    def forward(self, x, hyperedge_index, batch):
        xs = []
        for _ in range(self.num_convs):
            x = F.relu( self.convs[_](x, hyperedge_index))   
            xs.append(x)

        if self.readout == 'last':
            graph_emb = self.pool(xs[-1], batch)

        elif self.readout == 'concat':
            graph_emb = torch.cat([self.pool(x, batch) for x in xs], 1) 

        elif self.readout == 'add':
            graph_emb = 0
            for x in xs:
                graph_emb += self.pool(x, batch)

        return graph_emb, torch.cat(xs, 1)   
    
    
    def get_convs(self):
        convs = torch.nn.ModuleList()
        for i in range(self.num_convs):
            if i == 0:
                conv = HypergraphConv(self.num_node_features, self.nhid)
            else:
                conv = HypergraphConv(self.nhid, self.nhid)      
            convs.append(conv)

        return convs
    
    def get_pool(self):
        if self.pooling == 'add':
            pool = global_add_pool
        elif self.pooling == 'max':
            pool = global_max_pool
        elif self.pooling == 'mean':
            pool = global_mean_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(self.pooling))

        return pool
    

class Encoder_GCN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, pooling, readout):
        super(Encoder_GCN, self).__init__()

        self.num_gc_layers = num_gc_layers

        self.pooling = pooling
        self.readout = readout
        self.pool = self.get_pool()

        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                conv = GCNConv(dim, dim)
            else:
                conv = GCNConv(num_features, dim)
            self.convs.append(conv)


    def forward(self, x, edge_index, batch):
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        if self.readout == 'last':
            graph_emb = self.pool(xs[-1], batch)
        elif self.readout == 'concat':
            graph_emb = torch.cat([self.pool(x, batch) for x in xs], 1)
        elif self.readout == 'add':
            graph_emb = 0
            for x in xs:
                graph_emb += self.pool(x, batch)
  
        return graph_emb, torch.cat(xs, 1)   
    
    
    def get_pool(self):
        if self.pooling == 'add':
            pool = global_add_pool
        elif self.pooling == 'max':
            pool = global_max_pool
        elif self.pooling == 'mean':
            pool = global_mean_pool
        else:
            raise ValueError("Pooling Name <{}> is Unknown".format(self.pooling))
        return pool