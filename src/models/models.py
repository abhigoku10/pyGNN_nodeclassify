import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np 
import math

import os
import sys
sys.path.append(os.getcwd())

from ..layers.adjacency import adjacency_cora,adjacency_images
from ..layers.layers import GraphConvolution , GraphAttentionLayer
from ..layers.layers import SpGraphAttentionLayer
from .. layers.layers import GraphWaveletNeuralNetwork

from .. layers.layers import MeanAggregator,Encoder

from .. layers.layers import GrandLayer

from .. layers.layers import GATconv

from .. layers.layers import SAGEGCN


class GraphSAGE(nn.Module):

    def __init__(self, input_dim, hidden_dims=[64, 64],num_neighbors_list=[10, 10]):

        super(GraphSAGE, self).__init__()

        self.num_layers = len(num_neighbors_list)
        self.num_neighbors_list = num_neighbors_list

        # Define each level of neighbor node feature aggregation layerr
        self.gcn = nn.ModuleList()
        self.gcn.append(SAGEGCN(input_dim, hidden_dims[0], 'relu'))
        for  index  in  range ( 0 , len ( hidden_dims ) -  2 ):
            self.gcn.append(SAGEGCN(hidden_dims[index], hidden_dims[index + 1], 'relu'))

        self.gcn.append(SAGEGCN(hidden_dims[-2], hidden_dims[-1], None))

    def forward(self, node_feature_list):
        # Each layer of neighbor node feature list, the first tensor is the source node feature
        hidden = node_feature_list

        for  i  in  range ( self . num_layers ):
            # Record the output of each hidden layer network
            next_hidden = []

            # Every hidden layer network
            gcn = self.gcn[i]

            for hop in range(self.num_layers - i):
                # Source node characteristics of each order
                src_node_features = hidden[hop]

                # Features of neighbor nodes of each order
                src_node_num = len(src_node_features)
                neighbor_node_features = hidden[hop + 1].view((
                    src_node_num, self.num_neighbors_list[hop], -1
                ))

                # Aggregate neighbor node features and calculate source node hidden layer features
                new_hidden = gcn(src_node_features, neighbor_node_features)
                next_hidden.append(new_hidden)

            hidden = next_hidden

            # Source node logits as output
        output = hidden[0]
        return output

class gat_all(nn.Module):
    '''
    https://github.com/marblet/gat-pytorch

    '''
    def __init__(self,nfeat,nclass, nhid, nhead,alpha,dropout,nhead_out=1):
        super(gat_all,self).__init__()

        self.attentions = [GATconv(nfeat,nhid, dropout=dropout,alpha=alpha)for _ in range(nhead)]
        self.out_atts = [GATconv(nhid*nhead,nclass,dropout=dropout,alpha=alpha)for _ in range(nhead_out)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i),attention)

        for i,attention in enumerate(self.out_atts):
            self.add_module('out_attr{}'.format(i),attention)

        self.reset_parameters()

    def reset_parameters(self):
        for att in self.attentions:
            att.reset_parameters()
        for att in self.out_atts:
            att.reset_parameters()

    def forward(self,x,edge_list):
        x = torch.cat([att(x,edge_list)for att in self.attentions],dim=1)

        x = F.elu(x)

        x= torch.sum(torch.stack([att(x,edge_list)for att in self.out_atts]),dim=0)/ len(self.out_atts)

        return F.log_softmax(x,dim=1)

        
class grand(nn.Module):
    '''
    https://github.com/THUDM/GRAND
    '''
    def __init__(self,nfeat,nhid,nclass,input_droprate,hidden_droprate,is_cuda=True,use_bn=False):
        super(grand,self).__init__()

        self.layer1 = GrandLayer(nfeat,nhid)
        self.layer2 = GrandLayer(nhid,nclass)
        
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.is_cuda = is_cuda
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def forward(self,x):
        if self.use_bn:
            x= self.bn1(x)
        x= F.dropout(x,self.input_droprate,training=self.training)
        x= F.relu(self.layer1(x))
        if self.use_bn:
            x= self.bn2(x)
        x= F.dropout(x,self.hidden_droprate,training=self.training)
        x= self.layer2(x)
        return x 

class fastgcn(nn.Module):
    '''
    https://github.com/Gkunnan97/FastGCN_pytorch

    '''

    def __init__(self, nfeat,nhid,nclass,dropout,sampler):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.sampler = sampler
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x, adj):
        outputs1 = F.relu(self.gc1(x, adj[0]))
        outputs1 = F.dropout(outputs1, self.dropout, training=self.training)
        outputs2 = self.gc2(outputs1, adj[1])
        return F.log_softmax(outputs2, dim=1)

    # def sampling(self, features, adj,input_dim,layersize):
    #     return self.sampler.sampling(features, adj,input_dim,layersize)
    def sampling(self,v_indicies):
        return self.sampler.sampling(v_indicies)


class gwnn(nn.Module):
    def __init__(self, node_cnt,feature_dims,hidden_dims,output_dims,wavelets,wavelets_inv,dropout_rate):
        super(gwnn,self).__init__()
        self.node_cnt = node_cnt
        self.feature_dims = feature_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.wavelets = wavelets
        self.wavelets_inv = wavelets_inv
        self.dropout_rate = dropout_rate


        self.conv1 = GraphWaveletNeuralNetwork(self.node_cnt,self.feature_dims,self.hidden_dims,self.wavelets,self.wavelets_inv)

        self.conv2 = GraphWaveletNeuralNetwork(self.node_cnt,self.hidden_dims, self.output_dims,self.wavelets,self.wavelets_inv)


    def forward(self, input):
        output_1 = F.dropout(F.relu(self.conv1(input)),training=self.training,p=self.dropout_rate)
        output_2 = self.conv2(output_1)
        pred =  F.log_softmax(output_2,dim=1)
        return pred 



class graphsage_sup(nn.Module):

    def __init__(self, num_classes, enc):
        super(graphsage_sup, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        
        nn.init.xavier_normal_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

class gat(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(gat,self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat,nhid, dropout= dropout\
            , alpha= alpha, concat= True)for _ in range(nheads)]

        for i, attention in enumerate (self.attentions):
            self.add_module('attention_{}'.format(i),attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class sp_gat(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(sp_gat, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, nhid , dropout= dropout, alpha= alpha, \
            concat=True)for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_attr = SpGraphAttentionLayer(nhid * nheads, nclass,dropout= dropout,alpha=alpha ,\
            concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_attr(x, adj))
        return F.log_softmax(x, dim=1)

class gcn_tkipf(nn.Module):
    def __init__(self,nfeat, nhid, nclass, dropout):
        super(gcn_tkipf,self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class gcn_spectraledge(nn.Module):
    def __init__(self,img_size=28 ,pred_edge = True ):
        super(gcn_spectraledge,self).__init__()
        self.pred_edge = pred_edge
        N = img_size** 2
        self.fc =  nn.Linear(N,10,bias=False)
        col, row = np.meshgrid(np.arange(img_size), np.arange(img_size))
        coord = np.stack((col,row),axis=2).reshape(-1,2)
        coord = (coord - np.mean(coord, axis=0)) / (np.std(coord, axis=0) + 1e-5)
        coord = torch.from_numpy(coord).float()  # 784,2
        coord = torch.cat((coord.unsqueeze(0).repeat(N, 1,  1),
                                    coord.unsqueeze(1).repeat(1, N, 1)), dim=2)
            #coord = torch.abs(coord[:, :, [0, 1]] - coord[:, :, [2, 3]])
        self.pred_edge_fc = nn.Sequential(nn.Linear(4, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, 1),
                                    nn.Tanh())
        self.register_buffer('coord', coord)

    def forward(self, x):
        B = x.size(0)
        
        self.A = self.pred_edge_fc(self.coord).squeeze()
        avg_neighbor_features = (torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1),
                                 x.view(B, -1, 1)).view(B, -1))
                                 
        return self.fc(avg_neighbor_features)


class gcn_spectral(nn.Module):
    def __init__(self,args, img_size=28):
        super(gcn_spectral, self).__init__()
        N = img_size ** 2 
        self.fc = nn.Linear(N, 10, bias=False)

        # precompute adjacency matrix before training
        A = adjacency_images(img_size)
        self.register_buffer('A', A)

    def forward(self, x):
        B = x.size(0)
        avg_neighbor_features = (torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1),
                                 x.view(B, -1, 1)).view(B, -1))
        return self.fc(avg_neighbor_features)
        

