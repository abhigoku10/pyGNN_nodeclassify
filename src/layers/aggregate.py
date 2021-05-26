"""Defines the feature aggregation mode of neighbors
"""


import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.getcwd())


"""Sampling neighbors based on nodes
    According to the source node sampling a specified number of neighbor nodes, using sampling with replacement,
    When the number of neighbor nodes of a node is less than the number of samples, there are duplicate nodes in the sampling results
"""


import  numpy  as  np


def sampling(src_nodes, sample_num, neighbor_dict):
    """First-order sampling based on the source node
        Inputs:
        -------
        src_nodes: list or numpy array, source node list
        sample_num: int, the number of neighbor nodes to be sampled
        neighbor_dict: dict, the mapping table from node to its neighbor node
        Output:
        -------
        sampling_results: numpy array, list of nodes after sampling
    """

    sampling_results = []
    for node in src_nodes:
        # Sampling with replacement from the neighbors of the node
        sample = np.random.choice(neighbor_dict[node], size=(sample_num,))
        sampling_results.append(sample)

    return np.asarray(sampling_results).flatten()


def multihop_sampling(src_nodes, sample_nums, neighbor_dict):
    """ Multi-level sampling based on the source node
        Inputs:
        -------
        src_nodes: list or numpy array, source node list
        sample_nums: list of ints, the number of neighbor nodes to be sampled for each order
        neighbor_dict: dict, the mapping table from node to its neighbor node
        Output:
        -------
        sampling_results: list of numpy array, list of nodes after each order of sampling
    """

    sampling_results = [src_nodes]
    for k, hopk_num in enumerate(sample_nums):
        # Sample neighbors for each order
        hopk_sampling = sampling(sampling_results[k], hopk_num, neighbor_dict)
        sampling_results.append(hopk_sampling)

    return sampling_results

class NeighborAggregator(nn.Module):
    """Defines the feature aggregation mode of neighbors
    """

    def __init__(self, input_dim, output_dim, use_bias=True, aggr_method='mean'):
        """Defines the feature aggregation mode of neighbors
            Inputs:
            -------
            input_dim: int, input feature dimension
            output_dim: int, output feature dimension
            use_bias: boolean, whether to use bias
            aggr_method: string, aggregation method, optional'mean','sum','max'
        """

        super(NeighborAggregator, self).__init__()
        assert aggr_method in ['mean', 'sum', 'max']

        self.use_bias = use_bias
        self.aggr_method = aggr_method

        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.__init_parameters()

        return

    def __init_parameters(self):
        """Initialize weights and biases
        """

        nn.init.kaiming_normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

        return

    def forward(self, neighbor_features):
        """ node aggregation feedforward
            Input:
            ------
            neighbor_features: tensor in shape [num_nodes, input_dim],
                              Neighbor characteristics of nodes
            Output:
            -------
            neighbor_hidden: tensor in shape [num_nodes, output_dim],
                             Node features after aggregating neighbor features
        """

        # Aggregate neighbor features
        if self.aggr_method == 'mean':
            aggr_neighbor = neighbor_features.mean(dim=1)
        elif self.aggr_method == 'sum':
            aggr_neighbor = neighbor_features.sum(dim=1)
        else:  # self.aggr_method == 'max'
            aggr_neighbor = neighbor_features.max(dim=1)

        # Linear transformation to obtain hidden layer features
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden