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
