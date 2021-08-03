import pdb
import math
import torch
import numpy as np
import scipy.sparse as sp
import sys
import os 
sys.path.append(os.getcwd())


from scipy.sparse.linalg import norm as sparse_norm
from torch.nn.parameter import Parameter



from ..utils.base_utils import sparse_mx_to_torch_sparse_tensor


class Sampler(object):
    def __init__ (self,features, adj,input_dim,layersize):
        
        self.input_dim = input_dim
        self.features = features
        self.adj = adj
        self.layersize = layersize
        self.num_layers = len(self.layersize)
        self.train_nodes_number = self.adj.shape[0]

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def sampling(self, v_indicies):
        raise NotImplementedError("sampling is not implimented")

    def _change_sparse_to_tensor(self, adjs):
        new_adjs = []
        for adj in adjs:
            new_adjs.append(sparse_mx_to_torch_sparse_tensor(adj).to(self.device))
        return new_adjs


class SamplerFastGCN(Sampler):
    def __init__(self,features, adj,input_dim,layersize):
        super(SamplerFastGCN,self).__init__(features, adj,input_dim,layersize)
        # NOTE: uniform sampling can also has the same performance!!!!
        # try, with the change: col_norm = np.ones(features.shape[0])
        # Calculate the importance of the target node (each column) for each source node (each row) (normalized)
        col_norm = sparse_norm(adj, axis=0)
        self.probs = col_norm / np.sum(col_norm)

    def sampling(self, v):
        """
        Inputs:
            v: batch nodes list
        """
        all_support = [[]] * self.num_layers

        cur_out_nodes = v
        for layer_index in range(self.num_layers-1, -1, -1):
            # Sampling nodes and adjacency matrix of each layer of network
            cur_sampled, cur_support = self._one_layer_sampling(
                cur_out_nodes,self.layersize[layer_index])

            
            all_support[layer_index] = cur_support
            cur_out_nodes = cur_sampled

        # Convert the adjacency matrix sampled in each layer to a sparse tensor
        all_support = self._change_sparse_to_tensor(all_support)
        sampled_X0 = self.features[cur_out_nodes]
        return sampled_X0, all_support, 0

    def _one_layer_sampling(self, v_indices, output_size):
        # NOTE: FastGCN described in paper samples neighboors without reference
        # to the v_indices. But in its tensorflow implementation, it has used
        # the v_indice to filter out the disconnected nodes. So the same thing
        # has been done here.
        # The adjacency matrix of the sampled source node, using the corresponding row of the initial adjacency matrix

        support = self.adj[v_indices, :]
        # Calculate the importance of all available target nodes of the source node
        neis = np.nonzero(np.sum(support, axis=0))[1]
        p1 = self.probs[neis]
        p1 = p1 / np.sum(p1)
        # Sampling the target node by importance
        sampled = np.random.choice(np.array(np.arange(np.size(neis))),
                                   output_size, True, p1)

        # Get the target node index list after sampling
        u_sampled = neis[sampled]
        # Get the sampled adjacency matrix composed of the source node and the target node
        support = support[:, u_sampled]
        # Normalization of adjacency matrix
        sampled_p1 = p1[sampled]

        support = support.dot(sp.diags(1.0 / (sampled_p1 * output_size)))
        return u_sampled, support


# if __name__ == '__main__':

#     datasetpath = "E:\\Freelance_projects\\GNN\\Tuts\\GNN_Tuts\\data\\cora_pubmend_citeseer_sem\\"
#     dataset = "cora"
#     coradata = Sem_Dataload(datasetpath,dataset,adj_train=True)
#     coradata.load_semdata()

#     adj = (getattr(coradata, dataset+'_norm_adjt'))
#     features = (getattr(coradata, dataset+'_features'))
#     adj_train = (getattr(coradata, dataset+'_norm_adj_train'))
#     train_features = (getattr(coradata, dataset+'_train_features'))
#     y_train = (getattr(coradata, dataset+'_y_train'))
#     y_test = (getattr(coradata, dataset+'_y_test'))
#     train_index = (getattr(coradata, dataset+'_train_index'))
#     val_index = (getattr(coradata, dataset+'_val_index'))
#     test_index = (getattr(coradata, dataset+'_test_index'))


#     # adj, features, adj_train, train_features, y_train, y_test, test_index = \
#     #     load_data("cora")
#     batchsize = 32
#     layer_sizes = [128, 128, batchsize]
#     input_dim = features.shape[1]

#     sampler = SamplerFastGCN(train_features, adj_train,
#                             input_dim=input_dim,
#                             layersize=layer_sizes)

#     batch_inds = list(range(batchsize))
#     sampled_feats, sampled_adjs, var_loss = sampler.sampling(batch_inds)
#     print(sampled_feats)
