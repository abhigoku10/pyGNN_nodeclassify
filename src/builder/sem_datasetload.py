import numpy as np
import scipy.sparse as sp
import math
import pickle as pkl
import pickle
from sklearn.preprocessing import normalize
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.special import iv
from scipy.integrate import quad
import sys
import os
sys.path.append(os.getcwd())

import networkx as nx

import torch
from torch.utils.data import DataLoader, Dataset
from utils.utils import nontuple_preprocess_features,normalize_adj,nontuple_preprocess_adj
from utils.utils import sparse_mx_to_torch_sparse_tensor


'''

Dataset Cora : https://github.com/Gkunnan97/FastGCN_pytorch
https://github.com/LeeWooJung/GCN_reproduce

Dataset pubmed : https://github.com/Gkunnan97/FastGCN_pytorch

Dataset siteseer : https://github.com/Gkunnan97/FastGCN_pytorch
'''
class Sem_Dataload(Dataset):
    def __init__(self, datapath,dataset,semisup_cora=False,adj_train= False ):
        super(Sem_Dataload,self).__init__()
        self.dataset_str = dataset
        self.datadir = datapath
        self.adj_train = adj_train
        self.semisup_cora= semisup_cora

    def parse_index_file(self,filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def sample_mask(self,idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    def process_features(self,features):
        row_sum_diag = np.sum(features, axis=1)
        row_sum_diag_inv = np.power(row_sum_diag, -1)
        row_sum_diag_inv[np.isinf(row_sum_diag_inv)] = 0.
        row_sum_inv = np.diag(row_sum_diag_inv)
        return np.dot(row_sum_inv, features)

    def load_semdata(self):
        """
        Loads input data from gcn/data directory
        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
        All objects above must be saved using python pickle module.
        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
        dataset_str = self.dataset_str
        datapath = self.datadir

        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("{}ind.{}.{}".format(datapath,dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        # test indices
        test_idx_reorder = self.parse_index_file("{}ind.{}.test.index".format(datapath,dataset_str))
        # sort the test index
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]


        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))#.astype(np.float32)

        ###To obtain degree 
        degree = np.sum(adj, axis=1)



        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        
     
        classes_num = labels.shape[1]

        ###Checking if missing any nodes 
        isolated_list = [i for i in range(labels.shape[0]) if np.all(labels[i] == 0)]
        if isolated_list:
            print(f"Warning: Dataset '{dataset_str}' contains {len(isolated_list)} isolated nodes")
        

        # Labelgeneration - #pytorch tags do not need to be one-hot encoded
        # my_labels = np.where(labels==1)[1]

        if (self.semisup_cora == True) and (self.adj_train==False) :
            ### For GCN 
            # https://github.com/LeeWooJung/GCN_reproduce
            features = nontuple_preprocess_features(features)
            features = torch.FloatTensor(features.todense())

            adj = adj + sp.eye(adj.shape[0]) # Add self connections
            adj = nontuple_preprocess_features(adj)
            adj = sparse_mx_to_torch_sparse_tensor(adj)

            labels = torch.LongTensor(labels)

            train_idx = range(x.shape[0])
            val_idx = range(x.shape[0], x.shape[0]+500)

            train_idx = torch.LongTensor(train_idx)
            val_idx = torch.LongTensor(val_idx)
            test_idx = torch.LongTensor(test_idx_range)

            setattr(self, dataset_str+'_adjlist'    , adj)
            setattr(self, dataset_str+'_features'   , features)
            setattr(self, dataset_str+'_label'     , labels)
            setattr(self, dataset_str+'_train_idx'       , train_idx)
            setattr(self, dataset_str+'_val_idx'      , val_idx)
            setattr(self, dataset_str+'_test_idx', test_idx)


        if self.adj_train == False and self.semisup_cora==False : 
        
            labels = np.array([np.argmax(row) for row in labels], dtype=np.long)  
            idx_test = test_idx_range.tolist()     
            idx_train = range(len(y))
            idx_val = range(len(y), len(y)+500)

        


            #### # Labelgeneration - TO obtain label implementation 
            ##https://github.com/taishan1994/pytorch_gat/blob/master/utils.py
            # train_my_labels_mask = self.sample_mask(idx_train, my_labels.shape[0])
            # val_my_labels_mask =  self.sample_mask(idx_val, my_labels.shape[0])
            # test_my_labels_mask =  self.sample_mask(idx_test, my_labels.shape[0])
            # train_my_labels =  self.my_labels[train_my_labels_mask]
            # val_my_labels =  self.my_labels[val_my_labels_mask]
            # test_my_labels =  self.my_labels[test_my_labels_mask]

            # print ( " Number of training nodes:" , len ( train_my_labels ))
            # print ( " Number of verification nodes:" , len ( val_my_labels ))
            # print ( " Number of test nodes:" , len ( test_my_labels ))


            train_mask = self.sample_mask(idx_train, labels.shape[0])
            val_mask = self.sample_mask(idx_val, labels.shape[0])
            test_mask = self.sample_mask(idx_test, labels.shape[0])

            y_train = labels[train_mask]
            y_val = labels[val_mask]
            y_test = labels[test_mask]

        

            # y_train[train_mask, :] = labels[train_mask, :]
            # y_val[val_mask, :] = labels[val_mask, :]
            # y_test[test_mask, :] = labels[test_mask, :]

            setattr(self, dataset_str+'_adjlist'    , adj)
            setattr(self, dataset_str+'_features'   , features)
            setattr(self, dataset_str+'_ytrain'     , y_train)
            setattr(self, dataset_str+'_yval'       , y_val)
            setattr(self, dataset_str+'_ytest'      , y_test)
            setattr(self, dataset_str+'_classes_num', classes_num)
            setattr(self, dataset_str+'_trainmask'  , train_mask)
            setattr(self, dataset_str+'_valmask'    , val_mask)
            setattr(self, dataset_str+'_testmask'   , test_mask)

        if self.adj_train == True and self.semisup_cora == False : 

            print("Load full supervised task.")
            idx_test = test_idx_range.tolist()

            idx_train = range(len(ally)-500)
            idx_val = range(len(ally)-500, len(ally))

            train_mask = self.sample_mask(idx_train, labels.shape[0])
            val_mask = self.sample_mask(idx_val, labels.shape[0])
            test_mask = self.sample_mask(idx_test, labels.shape[0])

            y_train = np.zeros(labels.shape)
            y_val = np.zeros(labels.shape)
            y_test = np.zeros(labels.shape)

            y_train[train_mask, :] = labels[train_mask, :]
            y_val[val_mask, :] = labels[val_mask, :]
            y_test[test_mask, :] = labels[test_mask, :]


            train_index = np.where(train_mask)[0]
            adj_train = adj[train_index, :][:, train_index]
            y_train = y_train[train_index]
            val_index = np.where(val_mask)[0]
            y_val = y_val[val_index]
            test_index = np.where(test_mask)[0]
            y_test = y_test[test_index]

            num_train = adj_train.shape[0]

            features = nontuple_preprocess_features(features).todense()
            train_features = features[train_index]

            norm_adj_train = nontuple_preprocess_adj(adj_train)
            norm_adj = nontuple_preprocess_adj(adj)



            setattr(self, dataset_str+'_norm_adjt'    , norm_adj)
            setattr(self, dataset_str+'_features'   , features)
            setattr(self, dataset_str+'_norm_adj_train'     , norm_adj_train)
            setattr(self, dataset_str+'_train_features'       , train_features)
            setattr(self, dataset_str+'_y_train'      , y_train)
            setattr(self, dataset_str+'_y_test', y_test)
            setattr(self, dataset_str+'_train_index'  , train_index)
            setattr(self, dataset_str+'_val_index'  , val_index)
            setattr(self, dataset_str+'_test_index'  , test_index)
            setattr(self, dataset_str+'_y_val'       , y_val)
 

    def load_all(self):
        ## get data
        data_path = self.datadir
        dataset = self.dataset_str
        suffixs = ['x', 'y', 'allx', 'ally', 'tx', 'ty', 'graph']
        objects = []
        for suffix in suffixs:
            file = os.path.join(data_path, 'ind.%s.%s'%(dataset, suffix))
            objects.append(pickle.load(open(file, 'rb'), encoding='latin1'))
        x, y, allx, ally, tx, ty, graph = objects
        x, allx, tx = x.toarray(), allx.toarray(), tx.toarray()

        # test indices
        test_index_file = os.path.join(data_path, 'ind.%s.test.index'%dataset)
        with open(test_index_file, 'r') as f:
            lines = f.readlines()
        indices = [int(line.strip()) for line in lines]
        min_index, max_index = min(indices), max(indices)

        # preprocess test indices and combine all data
        tx_extend = np.zeros((max_index - min_index + 1, tx.shape[1]))
        features = np.vstack([allx, tx_extend])
        features[indices] = tx
        ty_extend = np.zeros((max_index - min_index + 1, ty.shape[1]))
        labels = np.vstack([ally, ty_extend])
        labels[indices] = ty

        # get adjacency matrix
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))#.toarray()

        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)
        idx_test = indices

        train_mask = self.sample_mask(idx_train, labels.shape[0])
        val_mask = self.sample_mask(idx_val, labels.shape[0])
        test_mask = self.sample_mask(idx_test, labels.shape[0])
        zeros = np.zeros(labels.shape)
        y_train = zeros.copy()
        y_val = zeros.copy()
        y_test = zeros.copy()
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]
        # features = torch.from_numpy(self.process_features(features))
        # y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        #     torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test), \
        #     torch.from_numpy(train_mask), torch.from_numpy(val_mask), torch.from_numpy(test_mask)


        features = nontuple_preprocess_features(features).todense()
        
        
        setattr(self, dataset+'_adjlist'    , adj)
        setattr(self, dataset+'_features'   , features)
        setattr(self, dataset+'_ytrain'     , y_train)
        setattr(self, dataset+'_yval'       , y_val)
        setattr(self, dataset+'_ytest'      , y_test)
        setattr(self, dataset+'_trainmask'  , train_mask)
        setattr(self, dataset+'_valmask'    , val_mask)
        setattr(self, dataset+'_testmask'   , test_mask)


if __name__ == "__main__":

    datasetpath = "E:\\Freelance_projects\\GNN\\Tuts\\GNN_Tuts\\data\\cora_pubmend_citeseer_sem\\"
    dataset = "citeseer"
    coradata = Sem_Dataload(datasetpath,dataset,adj_train=False)
    coradata.load_semdata()
    features = (getattr(coradata, dataset+'_features'))#.to(device)
    print(features)
    # #Number of nodes
    # nb_nodes = features.shape[0]
    # #Feature Dimensions
    # ft_sizes = features.shape[1]
    # #Number of categories
    # nb_classes = my_labels.shape[0]

    coradata = Sem_Dataload(datasetpath,dataset,adj_train=False)
    coradata.load_all()
    features = (getattr(coradata, dataset+'_features'))#.to(device)
    print(features)