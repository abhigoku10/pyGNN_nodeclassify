GNN-Pytorch
Pytorch implementation of GNN methods and models. Pytorch implementation of GNN.

THe intention of this repo is understand and integrate the methods then obtaining highger results.

<br/>

## Node Classification

List of data sets used, OGB data sets can be used if conditions permit :

|    Dataset    | Nodes | Edges | Node Attr. | Classes | Train | Valid | Test |
| :-----------: | :---: | :---: | :--------: | :-----: | :---: | :---: | :--: |
|     Cora      | 2708  | 5429  |    1433    |    7    |  140  |  500  | 1000 |
|   Cora-Full   | 2708  | 5429  |    1433    |    7    | 1208  |  500  | 1000 |
|   Citeseer    | 3327  | 4732  |    3703    |    6    |  120  |  500  | 1000 |
| Citeseer-Full | 3327  | 4732  |    3703    |    6    | 1827  |  500  | 1000 |
|    Pubmed     | 19717 | 44338 |    500     |    3    |  60   |  500  | 1000 |
|  Pubmed-Full  | 19717 | 44338 |    500     |    3    | 18217 |  500  | 1000 |


|        Status        |             Method        |                             Paper                             | Cora  | Citeseer | Pubmed |
| :----------------: | :---------------------------: | :----------------------------------------------------------: | :---: | :------: | :------: |
| :heavy_check_mark: |       [GCN](./Node/GCN)       | [Kipf and Welling, 2017](https://arxiv.org/pdf/1609.02907.pdf) | 0.819 |  0.702   | 0.790 |
| :heavy_check_mark: | [GraphSAGE](./Node/GraphSAGE) | [Hamilton and Ying et al., 2017](https://arxiv.org/pdf/1706.02216.pdf) | 0.801 |  0.701   | 0.778 |
| :heavy_check_mark: |       [GAT](./Node/GAT)       | [Velickovic et al., 2018](https://arxiv.org/pdf/1710.10903.pdf) | 0.824 |  0.719  | 0.782 |
| :heavy_check_mark: | [FastGCN](./Node/FastGCN)<sup>**\***</sup> | [Chen and Ma et al., 2018](https://arxiv.org/pdf/1801.10247.pdf) | 0.854 | 0.779 | 0.855 |
| :heavy_check_mark: | [GRAND](./Node/GRAND) | [Feng and Zhang et al., 2020](https://arxiv.org/pdf/2005.11079.pdf) | 0.839 | 0.726 | 0.797 |




<br/>

##  Packages

| 依赖            | 版本   | 安装                                                         |
| --------------- | ------ | ------------------------------------------------------------ |
| python          | 3.8.6  | conda create --name gnn python=3.8.6                         |
| numpy           | 1.20.0 | pip install numpy==1.20.0                                    |
| scipy           | 1.6.0  | pip install scipy==1.6.0                                     |
| pyyaml          | 5.4.1  | pip install pyyaml==5.4.1                                    |
| scikit-learn    | 0.24.1 | pip install scikit-learn==0.24.1                             |
| pytorch         | 1.7.1  | conda install pytorch\==1.7.1 cudatoolkit=11.0 -c pytorch    |
| torch-geometric | 1.6.3  | [Installation](https://github.com/rusty1s/pytorch_geometric#installation) |
