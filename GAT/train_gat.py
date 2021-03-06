
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

from src.builder import graphdataload
from src. builder .graphdataload import classnames
from src. models.models import gat,sp_gat,gat_all
from src. utils.base_utils import  increment_path
from src. cfg.load_yaml import load_yamlcfg
from src. viz.viz_graph import t_SNE,plot_train_val_loss,plot_train_val_acc,pca_tsne,tsne_legend
from src. metrics.metric import classify,accuracy

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



#### Logging of the data into the txt file 
logging.getLogger().setLevel(logging.INFO)


def train(model, optimizer, features, edge_list, y_train,y_val,train_mask,val_mask,valmode):    
    
    model.train()
    optimizer.zero_grad()
    output = model(features,edge_list)


    loss_train = F.nll_loss(output[train_mask], y_train)
    acc_train = accuracy(output[train_mask], y_train)


    loss_train.backward()
    optimizer.step()

    if not  valmode:
        model.eval()
        with torch.no_grad():
            output = model(features, edge_list)

    loss_val = F.nll_loss(output[val_mask], y_val)
    acc_val = accuracy(output[val_mask], y_val)
 
    return loss_train.data.item(), acc_train.data.item() , loss_val.data.item(),acc_val.data.item()



def test(features,edge_list,test_mask,test_labels,model,data_type,outputviz,fig_path):

    model.eval()
    
    output = model(features, edge_list)

    loss_test = F.nll_loss(output[test_mask], test_labels)
    acc_test = accuracy(output[test_mask],test_labels)

    print("Test set results:","loss= {:.4f}".format(loss_test.data.item()),"accuracy= {:.4f}".format(acc_test.data.item()))
    logging.info("Testing loss: {:.4f} acc: {:.4f} ".format((loss_test.data.item()),(acc_test.data.item())))

    report = classify(output[test_mask],test_labels,classnames[data_type])    
    logging.info('GCN Classification Report: \n {}'.format(report))

    if outputviz :
        logging.info("\n[STEP 5]: Visualization {} results.".format(data_type))
        ## Make a copy for pca and tsneplot  
        outs = output[test_mask]
        label=test_labels
        # Calculate the predicted value

        # output format conversion
        output = output.cpu().detach().numpy()
        test_labels = test_labels.cpu().detach().numpy()

        # ground truth visualization
        gt_2d = t_SNE(output[test_mask], test_labels,2,fig_path)
        pca_tsne(outs,label,fig_path)
        tsne_legend(output[test_mask], test_labels, classnames[data_type], 'test_set',fig_path)

def main():
    parser = argparse.ArgumentParser(description="GAT /SP-GAT GNN Node Classification ")


    parser.add_argument('--config_path', action='store_true', \
    default='E:\\Freelance_projects\\GNN\\Tutsv2\\pyGNN_NC_XAI_V2\\GAT\\config\\gat_cora.yaml', help='Provide the config path')


    #### to create an inc of directory when running test and saving results 

    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    args = parser.parse_args()

    ###### Params loading from config File 

    config_path = args.config_path
    configs = load_yamlcfg(config_file= config_path)

    data_type = configs['Data']['datatype']
    data_saveresult= configs['Data']['save_results']
    train_datapath = configs['Data']['datapath']
    train_seedvalue = configs['random_state']
    train_modelsave = configs['Data']['model_save_path']
    train_savefig = configs['Data']['save_fig']
    test_outputviz = configs['Data']['output_viz']
    model_type=configs['gat']['type']

    #--------------------------------------------------------------#
    use_bn = configs['Model']['use_bn']
    model_hidden=configs['Model']['hidden_dim']
    model_droput=configs['Model']['dropout']
    model_nbheads= configs['Model']['nbheads']
    
    
    input_dim=configs['Model']['input_dim']
    output_dim=configs['Model']['output_dim']

    ###--------------------------------------------------------------#
    train_lr =  configs['Hyper']['LR']
    train_wtdecay = configs['Hyper']['weight_decay']
    train_epochs = configs['Hyper']['epochs']
    train_valmode = False
    train_patience=configs['Hyper']['Patience']
    model_alpha= configs['Hyper']['alpha']

    ###--------------------------------------------------------------#
    test_outputviz = configs['Logger']['output_viz']

    ### Creating an incremental Directories   
    save_dir = Path(increment_path(Path(data_saveresult) / model_type, exist_ok=args.exist_ok))  # increment run
    
    #### Creating and saving into the log file
    logsave_dir= "./"  
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logsave_dir+model_type + '_log.txt')),
                                logging.StreamHandler() ], 
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p'                       
                        )

    ####Bannering
    ascii_banner = pyfiglet.figlet_format("GAT !")
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

    if data_type == 'cora' or data_type == 'citeseer' or data_type =='pubmed':
                
        citedata = graphdataload.Graph_data(train_datapath,data_type,'SemiSupervised')
        citedata.load_data()

        adj = getattr(citedata, data_type+'_adjlist')
        features = getattr(citedata, data_type+'_features')
        y_train = getattr(citedata, data_type+'_ytrain')
        y_val = getattr(citedata, data_type+'_yval')
        y_test = getattr(citedata, data_type+'_ytest')
        n_class = getattr(citedata, data_type+'_classes_num')
        train_mask = getattr(citedata, data_type+'_trainmask')
        val_mask = getattr(citedata, data_type+'_valmask')
        test_mask = getattr(citedata, data_type+'_testmask')
        edge_list = getattr(citedata,data_type+'_edge_idx_loop')      

        logging.info("\n[STEP 1]: Processing {} dataset.".format(data_type))
        
        logging.info("| # of nodes : {}".format(adj.shape[0]))
        logging.info("| # of edges : {}".format(len(edge_list)))
        logging.info("| # of features : {}".format(features.shape[1]))
        logging.info("| # of clases   : {}".format(n_class))
        logging.info("| # of train set : {}".format(len(y_train)))
        logging.info("| # of val set   : {}".format(len(y_val)))
        logging.info("| # of test set  : {}".format(len(y_test))) 
        logging.info("| # of number of classes : {}".format(n_class))
        
    elif  data_type == 'pubmed' and model_type == 'gat':
        raise NotImplementedError(data_type)
    else:
        raise NotImplementedError(data_type)


    #######Data Loading is completed 

    ###Intialization of variables 
    num_feats = features.shape[1]
    num_class= n_class

    if model_type == 'gat':

        model = gat_all(nfeat=num_feats, 
                        nhid=model_hidden, 
                        nclass=num_class, 
                        dropout= model_droput, 
                        nhead=model_nbheads, 
                        alpha= model_alpha)
        optimizer = optim.Adam(model.parameters(),lr=train_lr, weight_decay=train_wtdecay)


        logging.info("\n[STEP 2]: Model {} definition.".format(model_type))
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
            edge_list = edge_list.cuda()
            y_train = y_train.to(device)
            y_val = y_val.to(device)
            y_test = y_test.to(device)


        t_total = time.time()
        bad_counter = 0
        best_epoch = 0
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []    
        loss_best = np.inf
        loss_mn = np.inf
        acc_best = 0.0
        acc_mx = 0.0
        best_epoch = 0

        logging.info("\n[STEP 3]: Model {} Training for epochs {}.".format(model_type,train_epochs))
        

        for epoch in range(train_epochs):

            to= time.time()

            train_loss,train_acc,val_loss,val_acc = train(model, optimizer, features,\
                edge_list, y_train,y_val,train_mask,val_mask,valmode= train_valmode)


            print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(train_loss),
                    'acc_train: {:.4f}'.format(train_acc),'loss_val: {:.4f}'.format(val_loss),
                    'acc_val: {:.4f}'.format(val_acc),'time: {:.4f}s'.format(time.time() - to))
            logging.info("Epoch:{:04d} loss_train:{:.4f} acc_train:{:.4f} loss_val:{:.4f} acc_val:{:.4f} time:{:.4f}s.".format((epoch+1),(train_loss),(train_acc),(val_loss),(val_acc),(time.time()-to)))

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

        loadpath = os.path.join(train_modelsave, '{}_{}.pkl'.format(model_type, best_epoch)) 
        model.load_state_dict(torch.load(loadpath))

        if test_outputviz: 
            (save_dir / 'test_fig' if test_outputviz else save_dir).mkdir(parents=True, exist_ok=True)
            testsave_fig = str(save_dir / 'test_fig')

        logging.info("\n[STEP 4]: Testing {} final model.".format(model_type))

        
        test(features,edge_list,test_mask,y_test,model,data_type,test_outputviz,testsave_fig)
    
    else:
        raise NotImplementedError(model_type)
if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()





