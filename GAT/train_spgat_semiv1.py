
#### Importing all the packages

from __future__ import print_function
from __future__ import division

### We import torch functionis 
import torch
from torchvision import datasets , transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import os
sys.path.append(os.getcwd())

#### We import custom functions 
from src. builder import graphdataload
from src. builder .graphdataload import classnames
from src. models.models import gat,sp_gat
from src. utils.utils import accuracy ,increment_path
from src. config.configy import load_config_data
from src. viz.viz_graph import t_SNE,plot_train_val_loss,plot_train_val_acc,pca_tsne,tsne_legend

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

### To train  dataset 
def train(model, optimizer, features, adj, labels, idx_train,idx_val,epoch,valmode):    
    
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)

    loss_train = F.nll_loss(output[idx_train], torch.max(labels[idx_train],1)[1])
    acc_train = accuracy(output[idx_train], torch.max(labels[idx_train],1)[1])

    loss_train.backward()
    optimizer.step()

    if not  valmode:       
        with torch.no_grad():
            model.eval()
            output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], torch.max(labels[idx_val],1)[1])
    acc_val = accuracy(output[idx_val], torch.max(labels[idx_val],1)[1])
 
    return loss_train.data.item(), acc_train.data.item() , loss_val.data.item(),acc_val.data.item()


### To Test the dataset 
def test(model, features, adj, idx_test, data_type,labels,outputviz,fig_path):

    model.eval()
    output = model(features, adj)

    loss_test = F.nll_loss(output[idx_test], torch.max(labels[idx_test],1)[1])
    acc_test = accuracy(output[idx_test],torch.max( labels[idx_test],1)[1])

    print("Test set results:","loss= {:.4f}".format(loss_test.data.item()),"accuracy= {:.4f}".format(acc_test.data.item()))
    logging.info("Testing loss: {:.4f} acc: {:.4f} ".format((loss_test.data.item()),(acc_test.data.item())))


    if outputviz :
        logging.info("\n[STEP 5]: Visualization {} results.".format(data_type))
        ## Make a copy for pca and tsneplot  
        outs = output
        label=labels

        output = output.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        # ground truth visualization
        result_all_2d = t_SNE(output, labels,2,fig_path)
        pca_tsne(outs,label,fig_path)
        # tsne_legend(output[idx_test], labels[idx_test], classnames[data_type], 'test_set',fig_path)


def main():

    parser = argparse.ArgumentParser(description="GAT /SP-GAT GNN Node Classification ")


    parser.add_argument('--config_path', action='store_true', \
    default='.\\config\\spgat_pubmed.yaml', help='Provide the config path')


    #### to create an inc of directory when running test and saving results 

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
    model_nbheads =  model_config['nb_heads']
    model_droput =  model_config['dropout']
    model_alpha =  model_config['alpha']
 
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
    save_dir = Path(increment_path(Path(data_saveresult) / model_type, exist_ok=args.exist_ok))  # increment run
    
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
    ascii_banner = pyfiglet.figlet_format("SP GAT !")
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
        class_num = getattr(citedata, data_type+'_classes_num')

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
    num_feats = features.shape[1]
    num_class= class_num

    if model_type == 'gat':

        model = gat(nfeat=num_feats,nhid= model_hidden,nclass=num_class, 
                    dropout=model_droput, 
                    nheads=model_nbheads, 
                    alpha= model_alpha)

        optimizer = optim.Adam(model.parameters(),lr=train_lr, weight_decay=train_wtdecay)

        logging.info("\n[STEP 2]: Model {} definition.".format(model_type))
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

        features, adj, labels = Variable(features), Variable(adj), Variable(labels)

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
        
    ############################################## Training Started ##############
        for epoch in range(train_epochs):

            to= time.time()
            train_loss,train_acc,val_loss,val_acc = train(model, optimizer, features,\
                 adj, labels, idx_train,idx_val,epoch,valmode= train_valmode)

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
        test(model, features, adj, idx_test,data_type, labels,test_outputviz,testsave_fig)
    
    ############################################## Testing Completed

    elif model_type == 'sp_gat':

        logging.info("\n[STEP 2]: Model {} definition.".format(model_type)) 
        model = sp_gat(nfeat=num_feats, 
                        nhid=model_hidden, 
                        nclass=num_class, 
                        dropout= model_droput, 
                        nheads=model_nbheads, 
                        alpha= model_alpha)

        optimizer = optim.Adam(model.parameters(),lr=train_lr, weight_decay=train_wtdecay)

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

        features, adj, labels = Variable(features), Variable(adj), Variable(labels)

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
        
    ############################################## Training Started ##############
        for epoch in range(train_epochs):

            to= time.time()
            train_loss,train_acc,val_loss,val_acc = train(model, optimizer, features,\
                 adj, labels, idx_train,idx_val,epoch,valmode= train_valmode)

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

        test(model, features, adj, idx_test, data_type,labels,test_outputviz,testsave_fig)

    ############################################## Testing Completed

    else:
        raise NotImplementedError(model_type)
if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()





