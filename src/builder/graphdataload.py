### Importing the basisc packages 

import numpy as np
import scipy.sparse as sp
import math
import pickle as pkl

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
print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))


from ..cfg.load_yaml import load_yamlcfg

from ..utils.base_utils import nontuple_preprocess_features,normalize_adj,nontuple_preprocess_adj
from ..utils.base_utils import sparse_mx_to_torch_sparse_tensor
from ..utils.base_utils import  adj_list_from_dict ,  add_self_loops
from ..viz.pyvis_viz import draw_graph3







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

        
        # g = nx.to_networkx_graph(graph)
        # draw_graph3(g,output_filename='graph_output.html', notebook=False)

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
        labels[test_idx_reorder, :] = labels[test_idx_range, :] # onehot        
     
        classes_num = labels.shape[1]

        ###Checking if missing any nodes 
        isolated_list = [i for i in range(labels.shape[0]) if np.all(labels[i] == 0)]


        if isolated_list: print(f"Warning: Dataset '{dataset_str}' contains {len(isolated_list)} isolated nodes")
        

        print("| # of nodes : {}".format(adj.shape[0]))
        print("| # of edges : {}".format(adj.sum().sum()/2))

        if (self.citation_type == 'SemiSupervised'):


            ##### Creating a variable to save the values before normalizing 
            adj_unnorm =  adj
            features_unnorm = features
            label_unnorm = labels
        

            print("*******Loading Semi-Supervised Data******")

            ### Creating the adjacency matrix with self loop connections 
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = adj + sp.eye(adj.shape[0]) # Add self connections

            ### Conversion of data to tuple representation 
            # https://github.com/LeeWooJung/GCN_reproduce
            features = nontuple_preprocess_features(features)
            adj = nontuple_preprocess_features(adj)

            ### Conversion of sparse data to torch tensor 
            adj = sparse_mx_to_torch_sparse_tensor(adj)

            

            

            ##### To obtain the indexes and base labels based on the y from data 
            train_idx = range(y.shape[0])
            val_idx = range(y.shape[0], y.shape[0]+500)

            ##### To obtain the indexes based on the mask logic  

            labels_ = np.array([np.argmax(row) for row in label_unnorm], dtype=np.long)  
            idx_test = test_idx_range.tolist()     
            idx_train = range(len(y))
            idx_val = range(len(y), len(y)+500)
            train_mask = self.sample_mask(idx_train, labels_.shape[0])
            val_mask = self.sample_mask(idx_val, labels_.shape[0])
            test_mask = self.sample_mask(idx_test, labels_.shape[0])
            y_train = labels_[train_mask]
            y_val = labels_[val_mask]
            y_test = labels_[test_mask]


            ### to obtain the edge details  of the graph
            edge_idx ,edge_weight = adj_list_from_dict(graph)
            edge_idx_loop = add_self_loops(edge_idx, features.shape[1])
            graph = nx.from_dict_of_lists(graph)
            edges = graph.edges()

            ##### Conversion of all the data into tensor format  
            ### Conversion of feature data to tensor format 
            features = torch.FloatTensor(features.todense())

            

            labels = torch.LongTensor(labels)
            train_idx = torch.LongTensor(train_idx)
            val_idx = torch.LongTensor(val_idx)
            test_idx = torch.LongTensor(test_idx_range)
            degree = torch.LongTensor(degree)

            y_train = (torch.from_numpy(y_train)).long()
            y_val = (torch.from_numpy(y_val)).long()
            y_test = (torch.from_numpy(y_test)).long()
            # train_mask = (torch.from_numpy(train_mask)).long()
            # val_mask = (torch.from_numpy(val_mask)).long()
            # test_mask = (torch.from_numpy(test_mask)).long()

            
            
            print("| # of train set : {}".format(len(train_idx)))
            print("| # of val set   : {}".format(len(val_idx)))
            print("| # of test set  : {}".format(len(test_idx)))
            print("| # of features : {}".format(features.shape[1]))
            print("| # of clases   : {}".format(classes_num))            
            print("| # of degree  : {}".format(len(degree)))
            print("| # of edges  : {}".format((edge_idx.shape)))
            print("| # of edges with loop  : {}".format((edge_idx_loop.shape)))



            setattr(self, dataset_str+'_adjlist'    , adj)
            setattr(self, dataset_str+'_features'   , features)
            setattr(self, dataset_str+'_train_idx'       , train_idx)
            setattr(self, dataset_str+'_val_idx'      , val_idx)
            setattr(self, dataset_str+'_test_idx', test_idx)
            setattr(self, dataset_str+'_degree'    , degree)


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
            setattr(self, dataset_str+'_labelsunorm'    , labels_)
            setattr(self, dataset_str+'_edge_weight'    , edge_weight)
            

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

    # datasetpath = "E:\\Freelance_projects\\GNN\\Tuts\\pyGNN\\GCN\\config\\gcn_cora.yaml"
    # dataset = "citeseer"
    # citation = 'SemiSuperv2'
    # coradata = Graph_data(datasetpath,dataset,citation)#adj_train=False)
    # coradata.load_data()

    configs = load_yamlcfg(config_file='E:\\Freelance_projects\\GNN\\Tutsv2\\pyGNN_NC_XAI_V2\\GCN\\config\\gcn_cora.yaml')

    train_datapath = configs['Data']['datapath']
    data_type = configs['Data']['datatype']
    citedata = Graph_data(train_datapath,data_type,'SemiSupervised')
    citedata.load_data()

    classnames = {
        'citeseer': ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI'],
        'cora': ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks', 'Probabilistc_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'],
        'pubmed': ['Diabetes_Mellitus_Experimental', 'Diabetes_Mellitus_Type_1', 'Diabetes_Mellitus_Type_2'],
    }


    adj = getattr(citedata, data_type+'_adjlist')
    features = getattr(citedata, data_type+'_features')
    y_train = getattr(citedata, data_type+'_ytrain')
    y_val = getattr(citedata, data_type+'_yval')
    y_test = getattr(citedata, data_type+'_ytest')
    train_mask = getattr(citedata, data_type+'_trainmask')
    val_mask = getattr(citedata, data_type+'_valmask')
    test_mask = getattr(citedata, data_type+'_testmask')  
    n_class = getattr(citedata,data_type+ '_classes_num')
    