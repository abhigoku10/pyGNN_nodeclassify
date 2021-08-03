### Importing the basisc packages 

import numpy as np
import scipy.sparse as sp
import math
import pickle as pkl
import pickle
from sklearn.preprocessing import normalize
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.special import iv
from scipy.integrate import quad
from scipy.sparse import coo_matrix
import sys
import os
sys.path.append(os.getcwd())

import networkx as nx
### We import torch functions 
import torch
from torch.utils.data import DataLoader, Dataset

### We import custom functions from other package

from ..utils.utils import nontuple_preprocess_features,normalize_adj,nontuple_preprocess_adj
from ..utils.utils import sparse_mx_to_torch_sparse_tensor
from ..utils.utils import  adj_list_from_dict ,  add_self_loops

classnames = {
    'citeseer': ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI'],
    'cora': ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistc_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'],
    'pubmed': ['Diabetes_Mellitus_Experimental', 'Diabetes_Mellitus_Type_1', 'Diabetes_Mellitus_Type_2'],
}

'''

Dataset Cora : https://github.com/Gkunnan97/FastGCN_pytorch
https://github.com/LeeWooJung/GCN_reproduce

Dataset pubmed : https://github.com/Gkunnan97/FastGCN_pytorch

Dataset siteseer : https://github.com/Gkunnan97/FastGCN_pytorch
'''
class Graph_data(Dataset):
    def __init__(self, datapath,dataset,citation='semi_1' ):
        super(Graph_data,self).__init__()
        self.dataset_str = dataset
        self.datadir = datapath
        self.citation_type = citation # 'semi_2' # 'full'

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

    

    def load_data(self):
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


        print("\n[STEP 1]: Processing {} dataset.".format(dataset_str))

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

        print("| # of nodes : {}".format(adj.shape[0]))
        print("| # of edges : {}".format(adj.sum().sum()/2))

        ###To obtain degree 
        degree = np.sum(adj, axis=1)



        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :] # onehot
        
     
        classes_num = labels.shape[1]

        ###Checking if missing any nodes 
        isolated_list = [i for i in range(labels.shape[0]) if np.all(labels[i] == 0)]
        if isolated_list:
            print(f"Warning: Dataset '{dataset_str}' contains {len(isolated_list)} isolated nodes")
        

        if (self.citation_type == 'SemiSuperv1'):
        

            print("Load semi-supervised task.")

            ### For GCN 
            # https://github.com/LeeWooJung/GCN_reproduce
            features = nontuple_preprocess_features(features)
            features = torch.FloatTensor(features.todense())

            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)###This was added for grand archi
            adj = adj + sp.eye(adj.shape[0]) # Add self connections

            
            adj = nontuple_preprocess_features(adj)
            adj = sparse_mx_to_torch_sparse_tensor(adj)





            labels = torch.LongTensor(labels)
            # idx_test = test_idx_range.tolist()
            # labels = torch.LongTensor(np.argmax(labels, -1))

            graph = nx.from_dict_of_lists(graph)
            edges = graph.edges()

            train_idx = range(y.shape[0])
            val_idx = range(y.shape[0], y.shape[0]+500)

            ##### do all the preprocessing of feature and adj here 

            train_idx = torch.LongTensor(train_idx)
            val_idx = torch.LongTensor(val_idx)
            test_idx = torch.LongTensor(test_idx_range)
            degree = torch.LongTensor(degree)

            print("| # of features : {}".format(features.shape[1]))
            print("| # of clases   : {}".format(ally.shape[1]))

            print("| # of train set : {}".format(len(train_idx)))
            print("| # of val set   : {}".format(len(val_idx)))
            print("| # of test set  : {}".format(len(test_idx)))

            setattr(self, dataset_str+'_adjlist'    , adj)
            setattr(self, dataset_str+'_features'   , features)
            setattr(self, dataset_str+'_label'     , labels)
            setattr(self, dataset_str+'_train_idx'       , train_idx)
            setattr(self, dataset_str+'_val_idx'      , val_idx)
            setattr(self, dataset_str+'_test_idx', test_idx)
            setattr(self, dataset_str+ '_classes_num',classes_num)
            setattr(self, dataset_str+'_edges'    , edges)
            setattr(self, dataset_str+'_degree'    , degree)
            

        elif self.citation_type == 'SemiSuperv2':
            
            print("Load semi-supervised task.")        
        
            labels = np.array([np.argmax(row) for row in labels], dtype=np.long)  
            idx_test = test_idx_range.tolist()     
            idx_train = range(len(y))
            idx_val = range(len(y), len(y)+500)

            ##### Creating a variable to save the values before normalizing 
            adj_unnorm =  adj
            features_unnorm = features
            

         

            ####Normalising the adjacency matrix and features 
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = adj + sp.eye(adj.shape[0]) # Add self connections
            adj = nontuple_preprocess_features(adj)
            

            features = nontuple_preprocess_features(features)           


            train_mask = self.sample_mask(idx_train, labels.shape[0])
            val_mask = self.sample_mask(idx_val, labels.shape[0])
            test_mask = self.sample_mask(idx_test, labels.shape[0])

            y_train = labels[train_mask]
            y_val = labels[val_mask]
            y_test = labels[test_mask]

            ##Passing the data to the tensor 
            features = torch.FloatTensor(features.todense())
            adj = sparse_mx_to_torch_sparse_tensor(adj)

            ###optional check 
            # idx_train = torch.LongTensor(idx_train)
            # idx_val = torch.LongTensor(idx_val)
            # idx_test = torch.LongTensor(idx_test)
            degree = torch.LongTensor(degree)

            ### to obtain the edges of the graph 
            edge_idx = adj_list_from_dict(graph)
            edge_idx_loop = add_self_loops(edge_idx, features.size(0))
            graph = nx.from_dict_of_lists(graph)
            edges = graph.edges()

            # labels = torch.LongTensor(labels)


            print("| # of features : {}".format(features.shape[1]))
            print("| # of clases   : {}".format(classes_num))
            print("| # of train set : {}".format(len(y_train)))
            print("| # of val set   : {}".format(len(y_val)))
            print("| # of test set  : {}".format(len(y_test)))
            print("| # of degree  : {}".format(len(degree)))
            print("| # of edges  : {}".format((edge_idx.shape)))


            setattr(self, dataset_str+'_adjlist'    , adj)
            setattr(self, dataset_str+'_features'   , features)
            setattr(self, dataset_str+'_ytrain'     , y_train)
            setattr(self, dataset_str+'_yval'       , y_val)
            setattr(self, dataset_str+'_ytest'      , y_test)
            setattr(self, dataset_str+'_classes_num', classes_num)
            setattr(self, dataset_str+'_trainmask'  , train_mask)
            setattr(self, dataset_str+'_valmask'    , val_mask)
            setattr(self, dataset_str+'_testmask'   , test_mask)
            setattr(self, dataset_str+'_adjunnorm'    , adj_unnorm)
            setattr(self, dataset_str+'_featuresunnorm'    , features_unnorm)
            setattr(self, dataset_str+'_edges'    , edges)
            setattr(self, dataset_str+'_edge_idx'    , edge_idx)
            setattr(self, dataset_str+'_edge_idx_loop'    , edge_idx_loop)
            setattr(self, dataset_str+'_labels'    , labels)

        elif self.citation_type == 'Fullsuper' :
        

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

            if dataset_str == 'pubmed':
                norm_adj = 1*sp.diags(np.ones(norm_adj.shape[0])) + norm_adj
                norm_adj_train = 1*sp.diags(np.ones(num_train)) + norm_adj_train

            print("| # of features : {}".format(features.shape[1]))
            print("| # of clases   : {}".format(classes_num))
            print("| # of train set : {}".format(len(train_index)))
            print("| # of val set   : {}".format(len(val_index)))
            print("| # of test set  : {}".format(len(test_index)))
            print("| # of degree  : {}".format(len(degree)))
            print("| # of adj matrix set : {}".format(norm_adj.shape))
            print("| # of norm adj matrix set   : {}".format(norm_adj_train.shape))



            setattr(self, dataset_str+'_norm_adj'    , norm_adj)
            setattr(self, dataset_str+'_features'   , features)
            setattr(self, dataset_str+'_norm_adj_train'     , norm_adj_train)
            setattr(self, dataset_str+'_train_features'       , train_features)
            setattr(self, dataset_str+'_y_train'      , y_train)
            setattr(self, dataset_str+'_y_test', y_test)
            setattr(self, dataset_str+'_train_index'  , train_index)
            setattr(self, dataset_str+'_val_index'  , val_index)
            setattr(self, dataset_str+'_test_index'  , test_index)
            setattr(self, dataset_str+'_y_val'       , y_val)
            setattr(self, dataset_str+'_classnum'       , classes_num)
 

    


if __name__ == "__main__":

    datasetpath = "E:\\Freelance_projects\\GNN\\Tuts\\pyGNN\\GCN\\config\\gcn_cora.yaml"
    dataset = "citeseer"
    citation = 'SemiSuperv2'
    coradata = Graph_data(datasetpath,dataset,citation)#adj_train=False)
    coradata.load_data()



    classnames = {
        'citeseer': ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI'],
        'cora': ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistc_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'],
        'pubmed': ['Diabetes_Mellitus_Experimental', 'Diabetes_Mellitus_Type_1', 'Diabetes_Mellitus_Type_2'],
    }


    features = (getattr(coradata, dataset+'_features'))#.to(device)
    print(features)
    adj =(getattr(coradata, dataset+'_adjlist'))
    edge_idx=(getattr(coradata, dataset+'_edgelist'))
    labels=(getattr(coradata, dataset+'_labels'))
    train_idx =(getattr(coradata, dataset+'_train_idx'))
    val_idx=(getattr(coradata, dataset+'_val_idx'))
    test_idx=(getattr(coradata, dataset+'_test_idx'))
    