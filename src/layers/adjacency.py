import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import numpy as np
from scipy.spatial.distance import cdist
import scipy.sparse as sp


def adjacency_cora(edges, labels):
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    return adj 




def adjacency_images(img_size):
    col, row = np.meshgrid(np.arange(img_size), np.arange(img_size))
    coord = np.stack((col, row), axis=2).reshape(-1, 2) / img_size
    dist = cdist(coord, coord)  
    sigma = 0.05 * np.pi
    
    # Below, I forgot to square dist to make it a Gaussian (not sure how important it can be for final results)
    A = np.exp(- dist / sigma ** 2)
    print('WARNING: try squaring the dist to make it a Gaussian')
        
    A[A < 0.01] = 0
    A = torch.from_numpy(A).float()

    # Normalization as per (Kipf & Welling, ICLR 2017)
    D = A.sum(1)  # nodes degree (N,)
    D_hat = (D + 1e-5) ** (-0.5)
    A_hat = D_hat.view(-1, 1) * A * D_hat.view(1, -1)  # N,N

    # Some additional trick I found to be useful
    A_hat[A_hat > 0.0001] = A_hat[A_hat > 0.0001] - 0.2

    print(A_hat[:10, :10])
    return A_hat
