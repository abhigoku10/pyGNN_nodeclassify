import numpy as np
import scipy.sparse as sp
import torch
import glob
import logging
import math
import os
import platform
import random
import re
import subprocess
import time
from pathlib import Path 
import sys
sys.path.append(os.getcwd())

import networkx as nx


###
from sklearn.preprocessing import normalize
from scipy.special import iv
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.optimize import minimize
from scipy.integrate import quad

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes)) [i,:] for i,c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),dtype= np.int32)
    return labels_onehot


def adj_list_from_dict(graph):
    G = nx.from_dict_of_lists(graph)
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    indices = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    return indices

def add_self_loops(edge_list, size):
    i = torch.arange(size, dtype=torch.int64).view(1, -1)
    self_loops = torch.cat((i, i), dim=0)
    edge_list = torch.cat((edge_list, self_loops), dim=1)
    return edge_list

##TODO rename the funciton 
# def normalize(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy( np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

##### Calculation of the accuracy of the model 
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


#### Function to check the incrementation of the folder 
def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

#####

def nontuple_preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    ep = 1e-10
    r_inv = np.power(rowsum + ep, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def nontuple_preprocess_adj(adj):
    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return adj_normalized.tocsr()

######
# def count_parameters(model):
# 	table = PrettyTable(["Modules", "Parameters"])
# 	total_params = 0
# 	for name, parameter in model.named_parameters():
#         if not parameter.requires_grad:continue
#         param = parameter.numel()
#         table.add_row([name, param])
#         total_params+=param
#         print(table)
#         print(f"Total Trainable Params: {total_params}")
#     return total_params


def get_databatches(train_ind, train_labels, batch_size=64, shuffle=True):
    """
    Inputs:
        train_ind: np.array
    """
    nums = train_ind.shape[0]
    if shuffle:
        np.random.shuffle(train_ind)
    i = 0
    while i < nums:
        cur_ind = train_ind[i:i + batch_size]
        cur_labels = train_labels[cur_ind]
        yield cur_ind, cur_labels
        i += batch_size


###### GWNN wavelet basis calculation 
class gwnn_waveletbais(object):
    def __init__(self):
        super(gwnn_waveletbais,self).__init__()

    def laplacian(self,W):
        """Return the Laplacian of the weight matrix."""
        # Degree matrix.
        d = W.sum(axis=0)
        # Laplacian matrix.
        d = 1 / np.sqrt(d)
        D = sp.diags(d.A.squeeze(), 0)
        I = sp.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

        assert type(L) is sp.csr.csr_matrix
        return L


    def fourier(self,L, algo='eigh', k=100):
        """Return the Fourier basis, i.e. the EVD of the Laplacian."""
        
        def sort(lamb, U):
            idx = lamb.argsort()
            return lamb[idx], U[:, idx]

        if algo is 'eig':
            lamb, U = np.linalg.eig(L.toarray())
            lamb, U = sort(lamb, U)
        elif algo is 'eigh':
            lamb, U = np.linalg.eigh(L.toarray())
            lamb, U = sort(lamb, U)
        elif algo is 'eigs':
            lamb, U = sp.linalg.eigs(L, k=k, which='SM')
            lamb, U = sort(lamb, U)
        elif algo is 'eigsh':
            lamb, U = sp.linalg.eigsh(L, k=k, which='SM')

        return lamb, U
        

    
    def weight_wavelet(self,s,lamb,U):
        s = s
        for i in range(len(lamb)):
            lamb[i] = math.exp(-lamb[i]*s)

        Weight = np.dot(np.dot(U, np.diag(lamb)),np.transpose(U))

        return Weight

    def weight_wavelet_inverse(self,s,lamb,U):
        s = s
        for i in range(len(lamb)):
            lamb[i] = math.exp(lamb[i] * s)

        Weight = np.dot(np.dot(U, np.diag(lamb)), np.transpose(U))

        return Weight


    def threshold_to_zero(self,mx, threshold):
        """Set value in a sparse matrix lower than
        threshold to zero. 
        
        Return the 'coo' format sparse matrix.
        Parameters
        ----------
        mx : array_like
            Sparse matrix.
        threshold : float
            Threshold parameter.
        """
        high_values_indexes = set(zip(*((np.abs(mx) >= threshold).nonzero())))
        nonzero_indexes = zip(*(mx.nonzero()))

        if not sp.isspmatrix_lil(mx):
            mx = mx.tolil()   

        for s in nonzero_indexes:
            if s not in high_values_indexes:
                mx[s] = 0.0
        mx = mx.tocoo()
        mx.eliminate_zeros()
        return mx
    
    def largest_lamb(self,L):
        lamb, U = sp.linalg.eigsh(L, k=1, which='LM')
        lamb = lamb[0]
        #print(lamb)
        return lamb
    
    def wavelet_basis(self,adj,s,threshold):

        L = self.laplacian(adj)

        print('Eigendecomposition start...')
        start = time.time()

        lamb, U = self.fourier(L)

        elapsed = (time.time() - start)
        print(f'Eigendecomposition complete, Time used: {elapsed:.6g}s')

        print('Calculating wavelet...')
        start = time.time()

        Weight = self.weight_wavelet(s,lamb,U)
        inverse_Weight = self.weight_wavelet_inverse(s,lamb,U)

        elapsed = (time.time() - start)
        print(f'Wavelet get, Time used: {elapsed:.6g}s')
        del U,lamb

        print('Threshold to zero...')
        start = time.time()

        Weight[Weight < threshold] = 0.0
        inverse_Weight[inverse_Weight < threshold] = 0.0

        elapsed = (time.time() - start)
        print(f'Threshold complete, Time used: {elapsed:.6g}s')

        print('L1 normalizing...')
        start = time.time()

        Weight = normalize(Weight, norm='l1', axis=1)
        inverse_Weight = normalize(inverse_Weight, norm='l1', axis=1)

        elapsed = (time.time() - start)
        print(f'L1 normalizing complete, Time used: {elapsed:.6g}s')

        Weight = sp.coo_matrix(Weight)
        inverse_Weight = sp.coo_matrix(inverse_Weight)

        t_k = (Weight, inverse_Weight)
        return t_k

    def fast_wavelet_basis(self,adj,s,threshold,m):
        L = self.laplacian(adj)
        lamb = self.largest_lamb(L)

        print('Calculating wavelet...')
        start = time.time()
        a = lamb / 2
        c = []
        inverse_c = []
        for i in range(m + 1):
            f = lambda x: np.cos(i * x) * np.exp(s * a * (np.cos(x) + 1))
            inverse_f = lambda x: np.cos(i * x) * np.exp(-s * a * (np.cos(x) + 1))

            f_res = 2 * np.exp(s * a) * iv(i, s * a)
            inverse_f_res = 2 * np.exp(-s * a) * iv(i, -s * a)
            
            # Compare with result of numerical computation
            print(f'Difference in order {i}: ')
            print(f'{f_res - quad(f, 0, np.pi)[0] * 2 / np.pi:.3g}')
            print(f'{inverse_f_res - quad(inverse_f, 0, np.pi)[0] * 2 / np.pi:.3g}')

            c.append(f_res)
            inverse_c.append(inverse_f_res)

        T = [sp.eye(adj.shape[0])]
        T.append((1. / a) * L - sp.eye(adj.shape[0]))

        temp = (2. / a) * (L - sp.eye(adj.shape[0]))

        for i in range(2, m + 1):
            T.append(temp.dot(T[i - 1]) - T[i - 2])

        Weight = c[0] / 2 * sp.eye(adj.shape[0])
        inverse_Weight = inverse_c[0] / 2 * sp.eye(adj.shape[0])

        for i in range(1, m + 1):
            Weight += c[i] * T[i]
            inverse_Weight += inverse_c[i] * T[i]

        elapsed = (time.time() - start)
        print(f'Wavelet get, Time used: {elapsed:.6g}s')

        Weight, inverse_Weight = Weight.tocoo(), inverse_Weight.tocoo()

        #print((Weight.dot(inverse_Weight)).toarray())
        print('Threshold to zero...')
        start = time.time()

        Weight = self.threshold_to_zero(Weight, threshold)
        inverse_Weight =self.threshold_to_zero(inverse_Weight, threshold)

        elapsed = (time.time() - start)
        print(f'Threshold complete, Time used: {elapsed:.6g}s')

        print('L1 normalizing...')
        start = time.time()

        Weight = normalize(Weight, norm='l1', axis=1)
        inverse_Weight = normalize(inverse_Weight, norm='l1', axis=1)

        elapsed = (time.time() - start)
        print(f'L1 normalizing complete, Time used: {elapsed:.6g}s')

        t_k = (Weight, inverse_Weight)
        return t_k