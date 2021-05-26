
#### Importing all the packages

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

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
from src.builder import graphdataload
from src.builder import build
from src.builder .graphdataload import classnames
from src.models.models import gcn_tkipf,gcn_spectraledge,gcn_spectral
from src.utils.utils import accuracy ,increment_path
from src.config.configy import load_config_data
from src.viz.viz_graph import t_SNE,plot_train_val_loss,plot_train_val_acc
from src.viz.viz_graph import pca_tsne
from src.metrics.metric import classify

#### We import default functions
import time 
import argparse
import numpy as np 
import glob  
import logging
import random
from pathlib import Path
import pyfiglet





'''
gcn_spectraledge    : https://github.com/bknyaz/examples -- working 

gcn_spectral: https://github.com/bknyaz/examples -- working 

gcn : https://github.com/tkipf/pygcn -- working 

gat : https://github.com/Diego999/pyGAT --  working 

sp_gat : https://github.com/Diego999/pyGAT -- working 

graph_sage_sup : https://github.com/dsgiitr/graph_nets -- working

train_gcn_gen : https://github.com/LeeWooJung/GCN_reproduce --working

'''


#### Logging of the data into the txt file 
logging.getLogger().setLevel(logging.INFO)

### To Train the MNIST dataset 
def train_mnist(args, model, device, train_loader,optimizer, epoch):
    print('Training of {} dataset...'.format(args.data))
    model.train()
    for  batch_idx, (data,target) in enumerate(train_loader):
        data , target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval ==0:
            print('Train_Epoch: {} [{}/{} ({:.0f}%)]\t \
                    Loss:{:.6f}'.format(epoch, batch_idx* len(data),len(train_loader.dataset),\
                        100.*batch_idx/ len(train_loader),loss.item()))
    


#### To test the MNIST dataset 
def test_mnist(args, model , device, test_loader):
    print('Testing of {} dataset...'.format(args.data))

    model.eval()
    test_loss =0 
    correct =0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss+= F.cross_entropy(output, target,reduction='sum').item()
            pred = output.argmax(dim =1 , keepdim= True)
            correct+= pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
    '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


## To Train the  dataset 

def train(model, optimizer, features, adj, labels, idx_train,idx_val,epoch,valmode):

    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
   
    loss_train = F.nll_loss(output[idx_train],torch.max( labels[idx_train],1)[1])
    
    acc_train = accuracy(output[idx_train],torch.max(labels[idx_train],1)[1])
    loss_train.backward()
    optimizer.step()

    if not valmode:
        with torch.no_grad():
            model.eval()
            output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val],torch.max( labels[idx_val],1)[1])
    acc_val = accuracy(output[idx_val],torch.max( labels[idx_val],1)[1])

    return loss_train.data.item(), acc_train.data.item() , loss_val.data.item(),acc_val.data.item()



### To test the  dataset 
def test(model, features, adj, idx_test,labels,data_type,outputviz,fig_path):

    model.eval()
    output = model(features, adj)

    loss_test = F.nll_loss(output[idx_test], torch.max(labels[idx_test],1)[1])
    acc_test = accuracy(output[idx_test], torch.max(labels[idx_test],1)[1])

    print("Test set results:","loss= {:.4f}".format(loss_test.item()),"accuracy= {:.4f}".format(acc_test.item()))
    logging.info("Testing loss: {:.4f} acc: {:.4f} ".format(loss_test.item(),acc_test.item())) #array([0, 1], dtype=int64)

    report = classify(output,labels,classnames[data_type])
    logging.info('GCN Classification Report: \n {}'.format(report))

    if outputviz :

        logging.info("\n[STEP 5]: Visualization {} results.".format(data_type))
        ## Make a copy for pca and tsneplot  
        preds = output
        test_label=labels

        output = output.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()        
        base_tsne = t_SNE(output, labels,2,fig_path)

        pca_tsne(preds,test_label,fig_path)
        # tsne_legend(outputs[testmask], test_labels, classnames[data_type], 'test_set',fig_path)


def main():


    parser = argparse.ArgumentParser(description="GNN architectures")

    # Logging of the training based on intervals  
    parser.add_argument('--log_interval', type=int, default=100,help='how many batches to wait before logging training status')



    parser.add_argument('--config_path', action='store_true', \
    default='E:\\Freelance_projects\\GNN\\Tuts\\pyGNN\\GCN\\config\\gcn_cora.yaml', help='Provide the config path')


    ####Creation of increment directory for result saving 

    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

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
    model_droput =  model_config['dropout']
    
 
    ###### Loading Training parameters 
    train_hypers = configs['train_params']
    train_epochs=   train_hypers['max_num_epochs']
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
    save_dir = Path(increment_path(Path(data_saveresult) / data_type, exist_ok=args.exist_ok))  # increment run
    

    #### Creating and saving into the log file
    logsave_dir= "./"  
    logging.basicConfig(level=logging.INFO,
                        handlers=[
                                logging.FileHandler(os.path.join(logsave_dir+model_type + '_log.txt')),
                                logging.StreamHandler()
                            ], 
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'                       
                        )

    ####Bannering
    ascii_banner = pyfiglet.figlet_format("GCN !")
    print(ascii_banner)
    logging.info(ascii_banner)
  
    ###### Checking the availability of cuda else cpu 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using: {device}')
    logging.info("Using seed {}.".format(train_seedvalue))
    
    

    ##### Intialization of the seed 
    np.random.seed(train_seedvalue)
    torch.manual_seed(train_seedvalue)
    if device.type == 'cuda' :
        torch.cuda.manual_seed(train_seedvalue)


   ##### Data loading of the dataset

    if data_type == 'mnist':
        train_data , test_data = build.load_data_mnist(train_batch_size)
   
    elif data_type == 'cora' or data_type == 'citeseer' or data_type == 'pubmed':
       
        citedata = graphdataload.Graph_data(train_datapath,data_type,'SemiSuperv1')
        citedata.load_data()        

        adj = getattr(citedata, data_type+'_adjlist')
        features = getattr(citedata, data_type+'_features')
        labels = getattr(citedata, data_type+'_label')
        idx_train = getattr(citedata, data_type+'_train_idx')
        idx_val = getattr(citedata, data_type+'_val_idx')
        idx_test = getattr(citedata, data_type+'_test_idx')

        logging.info("\n[STEP 1]: Processing {} dataset.".format(data_type))        
        logging.info("| # of nodes : {}".format(adj.shape[0]))       
        logging.info("| # of features : {}".format(features.shape[1]))
        logging.info("| # of clases   : {}".format(labels.shape[1]))
        logging.info("| # of train set : {}".format(len(idx_train)))
        logging.info("| # of val set   : {}".format(len(idx_val)))
        logging.info("| # of test set  : {}".format(len(idx_test)))
  
    else:
        raise NotImplementedError(data_type)

    #######Data Loading is completed 

    ###Intialization of variables
    
    num_feat= features.shape[1]
    num_class= labels.shape[1]

    ##### Defining the type of model to use 
    if model_type == 'gcn_spectral':
        model = gcn_spectral(args)
        model.to(device)
       
        #### Logging the details 
        logging.info(str("Model_Info"))
        trainable_params =np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()])
        print('Total number of trainable parameters: %d'% trainable_params )
        logging.info(f"Total number of trainable parameters: {trainable_params}")
        logging.info(f"Number of epochs: {train_epochs}")

        # defining the optimizer 
        optimizer = optim.SGD(model.parameters(), lr=train_lr, weight_decay=1e-1)

        for epoch in range(1, train_epochs + 1):
            train_mnist(args, model, device, train_data, optimizer, epoch)
            test_mnist(args, model, device, test_data)

    elif model_type == 'gcn_spectraledge':
        model = gcn_spectraledge(args)
        model.to(device)
        # defining the optimizer 
        optimizer = optim.SGD(model.parameters(), lr=train_lr, weight_decay=train_wtdecay)
        #### Logging the details 
        logging.info(str("Model_Info"))
        trainable_params =np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()])
        print('Total number of trainable parameters: %d'% trainable_params )
        logging.info(f"Total number of trainable parameters: {trainable_params}")
        logging.info(f"Number of epochs: {train_epochs}")

        for epoch in range(1, train_epochs + 1):
            train_mnist(args, model, device, train_data, optimizer, epoch)
            test_mnist(args, model, device, test_data)


    elif model_type == 'gcn_tkipf':

        logging.info("\n[STEP 2]: Model {} definition.".format(model_type))
       
        model = gcn_tkipf(nfeat=num_feat,
                            nhid=model_hidden,
                            nclass=num_class,
                            dropout=model_droput)

        optimizer = optim.Adam(model.parameters(),lr=train_lr, weight_decay=train_wtdecay)

        logging.info("Model Architecture Used {}.".format(model_type))   
        logging.info(str(model))
        tot_params = sum([np.prod(p.size()) for p in model.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")
        logging.info(f"Number of epochs: {train_epochs}")
        
        #### Transfering the data into the GPU
        if device.type == 'cuda':
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []
        t_total = time.time()

        logging.info("\n[STEP 3]: Model {} Training for epochs {}.".format(model_type,train_epochs))
        
    ############################################## Training Started ##############
        for epoch in range(train_epochs):
            to= time.time()
            train_loss,train_acc,val_loss,val_acc =train(model, optimizer, features, adj, labels, idx_train,idx_val,epoch,valmode= train_valmode)

            print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(train_loss),
                    'acc_train: {:.4f}'.format(train_acc),'loss_val: {:.4f}'.format(val_loss),
                    'acc_val: {:.4f}'.format(val_acc),'time: {:.4f}s'.format(time.time() - to))
            logging.info("Epoch:{:04d} loss_train:{:.4f} acc_train:{:.4f} loss_val:{:.4f} acc_val:{:.4f} time:{:.4f}s.".format((epoch+1),(train_loss),(train_acc),(val_loss),(val_acc),(time.time()-to)))

            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)        
        

        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        logging.info(f"Total Training Time :{(time.time() - t_total)}")

        if train_savefig:

            logging.info("\n[STEP 3a]: Saving the Plot of Model {} Training(loss/acc)vs Validation(loss/acc).".format(model_type))

            (save_dir / 'train_plot' if train_savefig else save_dir).mkdir(parents=True, exist_ok=True)
            save_path = str(save_dir / 'train_plot') 
            num_epochs = range(1, train_epochs + 1)
            plot_train_val_loss(num_epochs,train_loss_history,val_loss_history,save_path)
            plot_train_val_acc(num_epochs,train_acc_history,val_acc_history,save_path)

    ############################################## Training Completed ##############

    ############################################## Testing Started
        if test_outputviz: 
            (save_dir / 'test_fig' if test_outputviz else save_dir).mkdir(parents=True, exist_ok=True)
            testsave_fig = str(save_dir / 'test_fig')

        logging.info("\n[STEP 4]: Testing {} final model.".format(model_type))
        
        test(model, features, adj, idx_test,labels,data_type,test_outputviz,testsave_fig)
    ############################################## Testing Completed

    else:
        raise NotImplementedError(model_type)



if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()





