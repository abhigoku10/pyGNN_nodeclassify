

#### Importing all the packages

from __future__ import print_function
from __future__ import division

### We import torch functionis 
import torch
from torchvision import datasets , transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

import sys
import os
sys.path.append(os.getcwd())

#### We import custom functions 
from src. builder import graphdataload
from src. builder .graphdataload import classnames
from src. models.models import GraphSAGE,graphsage_sup
from src. utils.utils import accuracy ,increment_path
from src. config.configy import load_config_data
from src. viz.viz_graph import t_SNE,plot_train_val_loss,plot_train_val_acc
from src. viz.viz_graph import pca_tsne , tsne_legend
from src. metrics.metric import classify
from src. layers.aggregate import multihop_sampling
from src. layers.layers import MeanAggregator,Encoder

#### We import default functions
import time 
import argparse
import numpy as np 
import glob 
import os 
import logging
import random
from pathlib import Path
from sklearn.metrics import f1_score
from collections import defaultdict
import pyfiglet



#### Logging of the data into the txt file 
logging.getLogger().setLevel(logging.INFO)

def main():

    parser = argparse.ArgumentParser(description="GNN architectures")
    parser.add_argument('--config_path', action='store_true', \
    default='.\\config\\graphsage_citeseer.yaml', help='Provide the config path')


    #### to create an inc of directory when running test and saving results 
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    parser.add_argument('--neighlist', default=[20, 20], help='each stage of sampling neighbor nodes')
    parser.add_argument('--n_batchperepoch', default=20, help='of each epoch cycle of the number of batches')


    args = parser.parse_args()

    ###### Params loading from config File 
    config_path = args.config_path
    configs = load_config_data(config_path)

    ###### Loading Dataset configurations
    dataset_config = configs['dataset_params']
    data_type = dataset_config['dataset_type']
    data_saveresult = dataset_config['save_results']


    ###### Loading Training config
    train_dataloader_config = configs['train_data_loader']    
    train_batch_size = train_dataloader_config['batch_size']
    train_datapath =  train_dataloader_config['data_path']

    
    ###### Loading Model configurations 
    model_config = configs['model_params']
    model_type = model_config['model_architecture']
    model_hidden = model_config['hidden']
    model_nbheads =  model_config['nb_heads']
    model_droput =  model_config['dropout']
    model_alpha =  model_config['alpha']
    model_ipdim = model_config['input_dim']
 
    ###### Loading Training parameters 
    train_hypers = configs['train_params']
    train_modelsave = train_hypers['model_save_path']
    train_modelload = train_hypers['model_load_path']
    train_epochs=   train_hypers['max_num_epochs']
    train_patience = train_hypers['patience']
    train_lr =      train_hypers['lr_rate']
    train_seedvalue = train_hypers['seed']
    train_wtdecay =  train_hypers['weight_decay']
    train_valmode = train_hypers['validationmode']
    train_savefig = train_hypers['save_fig']
    train_savelog = train_hypers['save_log']


    ###### Loading Testing parameters
    test_params = configs['test_params']
    test_modelload = test_params['model_load_path']
    test_outputviz = test_params['output_viz']

    

    ### Creating an incremental Directories   
    save_dir = Path(increment_path(Path(data_saveresult) / 'exp', exist_ok=args.exist_ok))  # increment run
    
    #### Creating and saving into the log file
    logsave_dir= "./"  
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logsave_dir+model_type + '_log.txt')),
                                logging.StreamHandler() ], 
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'                       
                        )

    ####Bannering
    ascii_banner = pyfiglet.figlet_format("Graph Sage !")
    print(ascii_banner)
    logging.info(ascii_banner)
  
    ###### To check if cuda is available else use the cpu 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using: {device}')
    logging.info("Using seed {}.".format(train_seedvalue))    

    #### Initialize the manual seed from argument 
    np.random.seed(train_seedvalue)
    torch.manual_seed(train_seedvalue)
    if device.type == 'cuda' :
        torch.cuda.manual_seed(train_seedvalue)

    ###### Data loading based on the dataset 
    if data_type == 'cora' :

        logging.info("\n[STEP 1]: Processing {} dataset.".format(data_type))

        num_nodes = 2708
        num_feats = model_ipdim
        feat_data = np.zeros((num_nodes, num_feats))
        labels = np.empty((num_nodes,1), dtype=np.int64)
        node_map = {}
        label_map = {}
        with open(os.path.join(train_datapath,"cora.content"), "r") as f:
            for i , line  in enumerate(f):
                info = line.strip().split()
                feat_data[i,:]=[float(x) for x in info[1:-1]]
                node_map[info[0]]=i
                if not info[-1] in label_map:
                    label_map[info[-1]]=len(label_map)
                labels[i]=label_map[info[-1]]

        adj_list = defaultdict(set)
        with open(os.path.join(train_datapath,"cora.cites"), "r") as f:
            for i , line in enumerate(f):
                info= line.strip().strip()
                info = info.split('\t') ## added this line since the value was not mapping
                assert len(info) == 2
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
                adj_list[paper1].add(paper2)
                adj_list[paper2].add(paper1)
        assert len(feat_data) == len(labels) == len(adj_list)
        
        

        n_class= 7
        
        features = nn.Embedding(num_nodes, model_ipdim)
        features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
        agg1 = MeanAggregator(features, cuda=True)
        enc1 = Encoder(features, model_ipdim, model_hidden, adj_list, agg1, gcn=True, cuda=False)
        agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, model_hidden, adj_list, agg2,
                base_model=enc1, gcn=True, cuda=False)
        enc1.num_samples = 5
        enc2.num_samples = 5

        graphsage = graphsage_sup(n_class, enc2)

        rand_indices = np.random.permutation(num_nodes)
        test = rand_indices[:1000]
        val = rand_indices[1000:1500]
        train = list(rand_indices[1500:])

        optimizer = optim.Adam(graphsage.parameters(),lr=train_lr, weight_decay=train_wtdecay)

        logging.info("\n[STEP 2]: Model {} definition.".format(model_type))
        logging.info("Model Architecture Used {}.".format(graphsage))   
        logging.info(str(graphsage))
        tot_params = sum([np.prod(p.size()) for p in graphsage.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")
        logging.info(f"Number of epochs: {train_epochs}")        
        logging.info("\n[STEP 3]: Model {} Training for epochs {}.".format(model_type,train_epochs))

        times=[]

        for batch in range(train_epochs):

            batch_nodes = train[:256]
            random.shuffle(train)
            start_time = time.time()

            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, 
                    Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)

            print('Epoch: {:04d}'.format(batch+1),'loss_train: {:.4f}'.format(loss.data.item()),'time: {:.4f}s'.format(end_time-start_time))
            logging.info("Epoch:{:04d} loss_train:{:.4f}  time:{:.4f}s.".format((batch+1),(loss.data.item()),(end_time-start_time)))


        val_output = graphsage.forward(val) 
        print ("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
        print ("Average batch time:", np.mean(times))

        logging.info("\n[STEP 4]: Testing {} final model.".format(model_type))
        test_output= graphsage.forward(test)
        F1_score =f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")
        print ("Testing F1:", F1_score)
        logging.info("Testing F1:{}".format(F1_score ) )  
        print ("Average batch time:", np.mean(times))

        ### Converting into Torch format
        out_labels =   labels[test]
        out_preds = test_output.data.numpy().argmax(axis=1)

        report = classify(out_preds,out_labels,classnames[data_type])        
        logging.info('GCN Classification Report: \n {}'.format(report))

        logging.info("\n[STEP 5]: Visualization {} results.".format(data_type))

        if test_outputviz: 
        
            (save_dir / 'test_fig' if test_outputviz else save_dir).mkdir(parents=True, exist_ok=True)
            testsave_fig = str(save_dir / 'test_fig')

            out_labels = np.concatenate(out_labels)

            # gt_2d = t_SNE(out_preds.reshape(-1,1), out_labels,1,testsave_fig)
            # pca_tsne(out_preds.reshape(-1,1),out_labels,testsave_fig)

            # tsne_legend(out_preds.reshape(-1, 1), out_labels, classnames[data_type], 'test_set',testsave_fig)

    elif data_type == 'pubmed':
        logging.info("\n[STEP 1]: Processing {} dataset.".format(data_type))

        #hardcoded for simplicity...
        num_nodes = 19717
        num_feats = 500
        feat_data = np.zeros((num_nodes, num_feats))
        labels = np.empty((num_nodes, 1), dtype=np.int64)
        node_map = {}

        with open(os.path.join(train_datapath,"Pubmed-Diabetes.NODE.paper.tab"), "r") as fp:
            fp.readline()
            feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
            for i, line in enumerate(fp):
                info = line.split("\t")
                node_map[info[0]] = i
                labels[i] = int(info[1].split("=")[1])-1
                for word_info in info[2:-1]:
                    word_info = word_info.split("=")
                    feat_data[i][feat_map[word_info[0]]] = float(word_info[1])

        adj_lists = defaultdict(set)

        with open(os.path.join(train_datapath,"Pubmed-Diabetes.DIRECTED.cites.tab"), "r") as fp:
            fp.readline()
            fp.readline()
            for line in fp:
                info = line.strip().split("\t")
                paper1 = node_map[info[1].split(":")[1]]
                paper2 = node_map[info[-1].split(":")[1]]
                adj_lists[paper1].add(paper2)
                adj_lists[paper2].add(paper1)
        
        n_class=3
        

        features = nn.Embedding(num_nodes, num_feats)
        features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

        agg1 = MeanAggregator(features, cuda=True)
        enc1 = Encoder(features, num_feats, model_hidden, adj_lists, agg1, gcn=True, cuda=False)
        agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
        enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, model_hidden, adj_lists, agg2,
                base_model=enc1, gcn=True, cuda=False)


        enc1.num_samples = 10
        enc2.num_samples = 25

        graphsage = graphsage_sup(n_class, enc2)

        rand_indices = np.random.permutation(num_nodes)
        test = rand_indices[:1000]
        val = rand_indices[1000:1500]
        train = list(rand_indices[1500:])

        optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)


        logging.info("\n[STEP 2]: Model {} definition.".format(model_type))
        logging.info("Model Architecture Used {}.".format(graphsage))   
        logging.info(str(graphsage))
        tot_params = sum([np.prod(p.size()) for p in graphsage.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")
        logging.info(f"Number of epochs: {train_epochs}")        
        logging.info("\n[STEP 3]: Model {} Training for epochs {}.".format(model_type,train_epochs))

        times=[]
        
        for batch in range(train_epochs):

            batch_nodes = train[:1024]
            random.shuffle(train)
            start_time = time.time()

            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, 
                    Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)

            print('Epoch: {:04d}'.format(batch+1),'loss_train: {:.4f}'.format(loss.data.item()),'time: {:.4f}s'.format(end_time-start_time))
            logging.info("Epoch:{:04d} loss_train:{:.4f}  time:{:.4f}s.".format((batch+1),(loss.data.item()),(end_time-start_time)))


        val_output = graphsage.forward(val) 
        print ("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
        print ("Average batch time:", np.mean(times))

        logging.info("\n[STEP 4]: Testing {} final model.".format(model_type))
        test_output= graphsage.forward(test)
        F1_score =f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")
        print ("Testing F1:", F1_score)
        logging.info("Testing F1:{}".format(F1_score ) )  
        print ("Average batch time:", np.mean(times))

        
        out_labels =   labels[test]
        out_preds = test_output.data.numpy().argmax(axis=1)

        report = classify(out_preds,out_labels,classnames[data_type])        
        logging.info('GCN Classification Report: \n {}'.format(report))
        logging.info("\n[STEP 5]: Visualization {} results.".format(data_type))

        if test_outputviz: 
        
            (save_dir / 'test_fig' if test_outputviz else save_dir).mkdir(parents=True, exist_ok=True)
            testsave_fig = str(save_dir / 'test_fig')

            out_labels = np.concatenate(out_labels)

            tsne_legend(out_preds.reshape(-1, 1), out_labels, classnames[data_type], 'test_set',testsave_fig)


    else:
        raise NotImplementedError(data_type)

        

        
if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()





