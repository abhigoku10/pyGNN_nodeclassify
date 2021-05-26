#!/usr/bin/env python3
import numpy as np 
import matplotlib.pyplot as plt
import scipy.sparse as sp
import os 
import sys
sys.path.append(os.getcwd())
from sklearn.manifold import TSNE
np.set_printoptions(precision=4)


import torch 
import torchvision
from torchvision import datasets , transforms


from ..utils.utils import encode_onehot,normalize,sparse_mx_to_torch_sparse_tensor
from ..layers.adjacency import adjacency_images, adjacency_cora


# from viz.viz_graph import t_SNE,viz_mnist_2d,scatter_mnist,viz_mnist_tsne

from collections import defaultdict


def load_data_mnist(bs,dataset="MNIST"):
    """Loading the data )"""
    print('Loading {} dataset...'.format(dataset))

        ### Building a dataset for training and test 
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    t = transforms.Compose([transforms.ToTensor(),\
                            transforms.Normalize(mean=(0.137),std=(0.3081))])

    train_dataset = torchvision.datasets.MNIST('E:\\Freelance_projects\\GNN\\Tuts\\GNN_Tuts\\data\\mnist', train = True , download = True ,transform = t )
    train_data =  torch.utils.data.DataLoader(train_dataset, batch_size =bs, shuffle = True , **kwargs)

    test_dataset = torchvision.datasets.MNIST('E:\\Freelance_projects\\GNN\\Tuts\\GNN_Tuts\\data\\mnist', train= False , download= True , transform = t )
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle= False,** kwargs)


    return train_data, test_data



def load_data_cora(path="E:\\Freelance_projects\\GNN\\Tuts\\GNN_Tuts\\data\\cora\\", dataset="cora"):
    '''
    https://github.com/weiyangfb/PyTorchSparseGAT
    
    '''
    
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype= np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    ## build graph

    idx = np.array(idx_features_labels[:,0],dtype=np.int32)
    idx_map= {j:i for i , j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)

    adj =adjacency_cora(edges, labels)
    #normalize the values 
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_cora(path="E:\\Freelance_projects\\GNN\\Tuts\\GNN_Tuts\\data\\cora\\", dataset="cora"):
    n_nodes = 2708
    n_features = 1433
    feat_data = np.zeros((n_nodes, n_features))
    labels = np.empty((n_nodes,1),dtype= np.int64)
    node_map ={}
    label_map ={}
    with open(os.path.join(path,"cora.content"), "r") as f:
        for i , line  in enumerate(f):
            info = line.strip().split()
            feat_data[i,:]=[float(x) for x in info[1:-1]]
            node_map[info[0]]=i
            if not info[-1] in label_map:
                label_map[info[-1]]=len(label_map)
            labels[i]=label_map[info[-1]]

    adj_list = defaultdict(set)
    with open(os.path.join(path,"cora.cites"), "r") as f:
        for i , line in enumerate(f):
            info= line.strip().strip()
            info = info.split('\t') ## added this line since the value was not mapping
            assert len(info) == 2
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_list[paper1].add(paper2)
            adj_list[paper2].add(paper1)
    assert len(feat_data) == len(labels) == len(adj_list)
    return feat_data, labels, adj_list



if __name__ == "__main__":
    # trainloader, testloader = load_data_mnist(32)
    # # dataiter = iter(trainloader)
    # # images, labels = dataiter.next()
    # # imshow(torchvision.utils.make_grid(images))
    # show(trainloader)

    ### Testing of cora dataset loading 
    # load_cora()

    ### Visualization of MNIST using matplotlib
    global batch_size  
    batch_size = 128
    trainloader, testloader = load_data_mnist(batch_size)
    # viz_mnist_2d(trainloader)
    # viz_mnist_tsne(trainloader)
