
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
from src .configy import config
from src .builder import graphdataload
from src. builder .graphdataload import classnames
from src. utils.utils import accuracy ,increment_path ,sparse_mx_to_torch_sparse_tensor,sgc_precompute
from src. viz.viz_graph import t_SNE,plot_train_val_loss,plot_train_val_acc
from src. viz.viz_graph import pca_tsne,tsne_legend
from src. metrics.metric import classify
from src. models.models import SGC


#### We import default functions
import time 
import argparse
import numpy as np 
import glob 
import os 
import logging
import random
from pathlib import Path
import pyfiglet
from time import perf_counter



#### Logging of the data into the txt file 
logging.getLogger().setLevel(logging.INFO)



def train(model,train_features, train_labels,val_features, val_labels,optimizer):
    

    model.train()
    optimizer.zero_grad()
    output = model(train_features)
    # loss_train = F.cross_entropy(output, train_labels)
    loss_train = F.cross_entropy(output,torch.max(train_labels,1)[1])
    # acc_train = accuracy(output, train_labels)
    acc_train = accuracy(output, torch.max(train_labels,1)[1])
    loss_train.backward()
    optimizer.step()
  
    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val=accuracy(output, torch.max(val_labels,1)[1])
        loss_val = F.cross_entropy(output, torch.max(val_labels,1)[1])

    return loss_train.data.item(), acc_train.data.item() , loss_val.data.item(),acc_val.data.item()

def test(model, test_features, test_labels,outputviz,data_type,fig_path):
    model.eval()
    output = model(test_features)
    loss_test = F.cross_entropy(output,torch.max(test_labels,1)[1])
    acc_test = accuracy(output, torch.max(test_labels,1)[1])

    print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))
    logging.info("Testing loss: {:.4f} acc: {:.4f} ".format(loss_test.item(),acc_test.item())) #array([0, 1], dtype=int64)

    report = classify(output,test_labels,classnames[data_type])   
    logging.info('Node Classification Report: \n {}'.format(report))

    if outputviz :
        logging.info("\n[STEP 5]: Visualization {} results.".format(data_type))
        ## Make a copy for pca and tsneplot  
        outs = output
        label=test_labels
        # Calculate the predicted value
        
        output = output.cpu().detach().numpy()
        labels = test_labels.cpu().detach().numpy()
        
        ## visualization with normal tsne and pc 
        result_tsne = t_SNE(output, labels,2,fig_path)
        pca_tsne(outs,label,fig_path)
        # tsne_legend(outs, label, classnames[data_type], 'test_set',fig_path)
    


def main():
    parser = argparse.ArgumentParser(description="GNN Node Classification")


    parser.add_argument('--config_path', action='store_true', \
    default='E:\\Freelance_projects\\GNN\\Tutsv2\\pyGNN_NC\\SGC\\config\\sgc_cora.yaml', help='Provide the config path')

    #### to create an inc of directory when running test and saving results 
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')


    degree =2 ###need to add to the input 

    args = parser.parse_args()


    ###### Params loading from config File 

    config_path = args.config_path
    configs = config.load_config(config_file= config_path)

    data_type = configs['Data']['datatype']
    data_saveresult= configs['Data']['save_results']
    train_datapath = configs['Data']['datapath']
    train_seedvalue = configs['random_state']
    model_type = 'sgc'
    train_modelsave = configs['Data']['model_save_path']
    train_savefig = configs['Data']['save_fig']
    test_outputviz = configs['Data']['output_viz']

    #--------------------------------------------------------------#
    dropout= configs['Model']['dropout']
    ipdim= configs['Model']['input_dim']
    opdim= configs['Model']['output_dim']
    hiddim= configs['Model']['hidden_dim']
    ip_droprate = configs['Model']['dropout']
    hid_droprate = configs['Model']['dropout']
    use_bn =  configs['Model']['use_bn']
    #--------------------------------------------------------------#
    train_lr =  configs['Hyper']['LR']
    train_wtdecay = configs['Hyper']['weight_decay']
    train_epochs = configs['Hyper']['epochs']
    train_valmode = False
    train_patience=configs['Hyper']['Patience']


    
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
    ascii_banner = pyfiglet.figlet_format("SGC !")
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
 
    if data_type == 'cora' or data_type == 'citeseer' or data_type == 'pubmed':
        
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

    logging.info("Dataset Used {}.".format(data_type))


    #######Data Loading is completed 

    ###Intialization of variables
    nclass = labels.shape[1]
    num_feats = features.shape[1]

    # Model and optimizer
    logging.info("\n[STEP 2]: Model {} definition.".format(model_type))

    model = SGC(nfeat=num_feats,nclass=nclass)

    # optimizer 
    optimizer = optim.Adam(model.parameters(),lr=train_lr, weight_decay=train_wtdecay)

    #### Logging the details 
    logging.info("Model Architecture Used {}.".format(model_type))   
    logging.info(str(model))
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    logging.info(f"Number of epochs: {train_epochs}")

    if device.type == 'cuda':
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    features, precompute_time = sgc_precompute(features, adj,degree)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    bad_counter = 0
    loss_best = np.inf
    loss_mn = np.inf
    acc_best = 0.0
    acc_mx = 0.0
    best_epoch = 0

    # Train model
    t_total = time.time()
    logging.info("\n[STEP 3]: Model {} Training for epochs {}.".format(model_type,train_epochs))
    
    for epoch in range(train_epochs):
        to= time.time()

        train_loss,train_acc,val_loss,val_acc = train(model, features[idx_train], labels[idx_train], features[idx_val], labels[idx_val],optimizer)
        
        print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(train_loss),'acc_train: {:.4f}'.format(train_acc),'loss_val: {:.4f}'.format(val_loss),\
                'acc_val: {:.4f}'.format(val_acc),'time: {:.4f}s'.format(time.time() - to))
        logging.info("Epoch:{:04d} loss_train:{:.4f} acc_train:{:.4f} loss_val:{:.4f} acc_val:{:.4f} time:{:.4f}s.".format((epoch+1),\
                        (train_loss),(train_acc),(val_loss),(val_acc),(time.time()-to)))

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)        
    
        path = os.path.join(train_modelsave, '{}_{}.pkl'.format(model_type, epoch)) 
        if val_loss_history[-1] <= loss_mn or val_acc_history[-1] >= acc_mx:
            if val_loss_history[-1] <= loss_best:
                loss_best = val_loss_history[-1]
                acc_best = val_acc_history[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), path)

            loss_mn = np.min((val_loss_history[-1], loss_mn))
            acc_mx = np.max((val_acc_history[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == train_patience:
            print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
            print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
            train_epochs=epoch+1
            break

    for f in glob.glob(os.path.join(train_modelsave,'*.pkl')):
        epoch_nb = int(f.split(os.path.sep)[-1].split('_')[-1].split('.')[0])
        if epoch_nb < best_epoch:
                os.remove(f)

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    logging.info(f"Total Training Completed :{(time.time() - t_total)}")

    for f in glob.glob(os.path.join(train_modelsave,'*.pkl')):
        epoch_nb =int(f.split(os.path.sep)[-1].split('_')[-1].split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)

    if train_savefig: 
        logging.info("\n[STEP 3a]: Saving the Plot of Model {} Training(loss/acc)vs Validation(loss/acc).".format(model_type))

        (save_dir / 'train_plot' if train_savefig else save_dir).mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / 'train_plot') 
        num_epochs = range(1, train_epochs + 1)
        plot_train_val_loss(num_epochs,train_loss_history,val_loss_history,save_path)
        plot_train_val_acc(num_epochs,train_acc_history,val_acc_history,save_path)

    ############################################## Training Completed ##############
    ############################################## Testing Started
    print('Loading {}th epoch'.format(best_epoch))

    #### Loading the model with same saved format 
    loadpath = os.path.join(train_modelsave, '{}_{}.pkl'.format(model_type, best_epoch)) 
    model.load_state_dict(torch.load(loadpath))
    
    if test_outputviz: 
        (save_dir / 'test_fig' if test_outputviz else save_dir).mkdir(parents=True, exist_ok=True)
        testsave_fig = str(save_dir / 'test_fig')

    logging.info("\n[STEP 4]: Testing {} final model.".format(model_type))

    test(model, features[idx_test], labels[idx_test],test_outputviz,data_type,testsave_fig)

if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()