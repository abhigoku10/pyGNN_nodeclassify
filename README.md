## GNN-Pytorch
Pytorch implementation of GNN methods and models. Pytorch implementation of GNN.

THe intention of this repo is understand and integrate the methods then obtaining highger results.

<br/>

## Node Classification

List of data sets used, OGB (https://github.com/snap-stanford/ogb) data sets can be used if conditions permit :

|    Dataset    | Nodes | Edges | Node Attr. | Classes | Train | Valid | Test |
| :-----------: | :---: | :---: | :--------: | :-----: | :---: | :---: | :--: |
|     Cora      | 2708  | 5429  |    1433    |    7    |  140  |  500  | 1000 |
|   Cora-Full   | 2708  | 5429  |    1433    |    7    | 1208  |  500  | 1000 |
|   Citeseer    | 3327  | 4732  |    3703    |    6    |  120  |  500  | 1000 |
| Citeseer-Full | 3327  | 4732  |    3703    |    6    | 1827  |  500  | 1000 |
|    Pubmed     | 19717 | 44338 |    500     |    3    |  60   |  500  | 1000 |
|  Pubmed-Full  | 19717 | 44338 |    500     |    3    | 18217 |  500  | 1000 |

<br/>

## Accuracy Metrics of Node Classification

|        Status        |             Method        |                             Paper                             | Cora  | Citeseer | Pubmed |
| :----------------: | :---------------------------: | :----------------------------------------------------------: | :---: | :------: | :------: |
| :heavy_check_mark: |       [GCN](./GCN)       | [Kipf and Welling, 2017](https://arxiv.org/pdf/1609.02907.pdf) | 0.8220 |  0.6960   | 0.780 |
| :heavy_check_mark: | [GraphSAGE](./GraphSAGE) | [Hamilton and Ying et al., 2017](https://arxiv.org/pdf/1706.02216.pdf) | 0.850 |  NA   | 0.808 |
| :heavy_check_mark: |       [GAT](./GAT)       | [Velickovic et al., 2018](https://arxiv.org/pdf/1710.10903.pdf) | 0.7840 |  0.7060  | NA |
| :heavy_check_mark: |       [SP_GAT](./GAT)       | [Yang Ye, Shihao Ji et al., 2021](https://arxiv.org/pdf/1912.00552.pdf) | 0.8180 |  0.7070  | 0.7750 |
| :heavy_check_mark: | [FastGCN](./FastGCN)<sup>**\***</sup> | [Chen and Ma et al., 2018](https://arxiv.org/pdf/1801.10247.pdf) | 0.8240 | 0.7710 | 0.8780 |
| :heavy_check_mark: | [GRAND](./GRAND) | [Feng and Zhang et al., 2020](https://arxiv.org/pdf/2005.11079.pdf) | 0.839 | 0.726 | 0.797 |
| :heavy_check_mark: | [GWNN](./GWNN) | [Bingbing Xu, Huawei Shen, Qi Cao et al., 2019](https://arxiv.org/pdf/1904.07785v1.pdf) | 0.7990 | 0.7710 | NA |

**\*** NA is because was not able to run in my system 


<br/>

##  Packages

| Package         | Version| Installation                                              |
| --------------- | ------ | ------------------------------------------------------------ |
| python          | 3.8.6  | conda create --name gnn python=3.8.6                         |
| numpy           | 1.20.0 | pip install numpy==1.20.0                                    |
| scipy           | 1.6.0  | pip install scipy==1.6.0                                     |
| pyyaml          | 5.4.1  | pip install pyyaml==5.4.1                                    |
| scikit-learn    | 0.24.1 | pip install scikit-learn==0.24.1                             |
| pytorch         | 1.7.1  | conda install pytorch\==1.7.1 cudatoolkit=11.0 -c pytorch    |
| torch-geometric | 1.6.3  | [Installation](https://github.com/rusty1s/pytorch_geometric#installation) |

<br/>

## Reference Links

GCN : https://github.com/tkipf/pygcn 

GAT : https://github.com/Diego999/pyGAT  

SP_GAT : https://github.com/Diego999/pyGAT  

GraphSage : https://github.com/dsgiitr/graph_nets 

GWNN : https://github.com/Yanqi-Chen/GWNN   

FastGCN : https://github.com/Gkunnan97/FastGCN_pytorch 

GRAND : https://github.com/THUDM/GRAND 

GraphLime: https://github.com/WilliamCCHuang/GraphLIME.git


<br/>

## License
This is open source project collected from other open sources , feel free to give feedback and raise comments 