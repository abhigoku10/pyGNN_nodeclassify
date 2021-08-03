
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
from src. models.models import fastgcn

from src. utils.base_utils import increment_path ,get_databatches,sparse_mx_to_torch_sparse_tensor
from src. layers.sampler import SamplerFastGCN
from src.cfg.load_yaml import load_yamlcfg
from src. viz.viz_graph import t_SNE,plot_train_val_loss,plot_train_val_acc
from src. viz.viz_graph import pca_tsne,tsne_legend
from src. metrics.metric import classify ,accuracy

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


 
def train(train_ind, train_labels, batch_size,val_feats,val_adj,val_labels,model,optimizer,valmode): 
#  y_val,adj_train,input_dim,layer_sizes,epochs
    model.train()

    # Used to record the loss of all batches in each epoch
    epoch_losses = []
    for batch_inds, batch_labels in get_databatches(train_ind,train_labels,batch_size):
        # Get the characteristics of the sampled node and the adjacency matrix of the sample
        sampled_feats, sampled_adjs, var_loss = model.sampling(batch_inds)

        optimizer.zero_grad()
        # Model output
        output = model(sampled_feats, sampled_adjs)

        # Calculate the loss function
        loss_train = F.nll_loss(output, batch_labels) + 0.5 * var_loss
        epoch_losses.append(loss_train.item())
        acc_train = accuracy(output, batch_labels)
        # Backpropagation
        
        loss_train.backward()
        optimizer.step()
        # just return the train loss of the last train epoch

  
    if not  valmode: 
        with torch.no_grad():
            model.eval()
            outputs = model(val_feats, val_adj)
            loss_val = F.nll_loss(outputs, val_labels)            
            acc_val = accuracy(outputs, val_labels)

   
    return loss_train.data.item(), acc_train.data.item() , loss_val.data.item(),acc_val.data.item()



def test(test_adj, test_feats, test_labels,model,outputviz,fig_path,data_type):

    model.eval()
    outputs = model(test_feats, test_adj)   

    loss_test = F.nll_loss(outputs, test_labels)
    acc_test = accuracy(outputs, test_labels)

    print("Test set results:","loss= {:.4f}".format(loss_test.data.item()),"accuracy= {:.4f}".format(acc_test.data.item()))
    logging.info("Testing loss: {:.4f} acc: {:.4f} ".format((loss_test.data.item()),(acc_test.data.item())))

    report = classify(outputs,test_labels,classnames[data_type]) 
    logging.info('GCN Classification Report: \n {}'.format(report))

    if outputviz :

        logging.info("\n[STEP 5]: Visualization {} results.".format(data_type))
        outs = outputs
        label=test_labels

        # output format conversion
        outputs = outputs.cpu().detach().numpy()
        test_labels = test_labels.cpu().detach().numpy()

        ## visualization with normal tsne and pc
        # gt_2d = t_SNE(outputs, test_labels,2,fig_path)
        # pca_tsne(outs,label,fig_path)
        tsne_legend(outputs, test_labels, classnames[data_type], 'test_set',fig_path)




def main():

    parser = argparse.ArgumentParser(description="GNN architectures")
    parser.add_argument('--config_path', action='store_true',\
    default='E:\\Freelance_projects\\GNN\\Tutsv2\\pyGNN_NC_XAI_V2\\FastGCN\\config\\fastgcn_pubmed.yaml', help='Provide the config path')


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
    model_type=configs['gcn']['type']


    #--------------------------------------------------------------#
    use_bn =  configs['Model']['use_bn']
    model_hidden          =configs['Model']['hidden_dim']    
    model_droput=configs['Model']['dropout']
    
    
    #--------------------------------------------------------------#
    train_lr =  configs['Hyper']['LR']
    train_wtdecay = configs['Hyper']['weight_decay']
    train_epochs = configs['Hyper']['epochs']
    train_valmode = False
    train_patience=configs['Hyper']['Patience']
    train_batch_size = configs['Hyper']['batch_size']

   


    # ###### Loading Training config
    # train_dataloader_config = configs['train_data_loader']    
    # train_batch_size = train_dataloader_config['batch_size']

    # ##TODO : add the data path in dataloader section 
    # train_datapath =  train_dataloader_config['data_path']

    
    # ###### Loading Model configurations 
    # model_config = configs['model_params']
    # model_type = model_config['model_architecture']
    # model_hidden = model_config['hidden']
    # model_droput =  model_config['dropout']

 
    # ###### Loading Training parameters
    # train_hypers = configs['train_params']
    # train_modelsave = train_hypers['model_save_path']
    # train_epochs=   train_hypers['max_num_epochs']
    # train_patience = train_hypers['patience']
    # train_lr =      train_hypers['lr_rate']
    # train_seedvalue = train_hypers['seed']
    # train_wtdecay =  train_hypers['weight_decay']
    # train_valmode = train_hypers['validationmode']
    # train_savefig = train_hypers['save_fig']
    # train_savelog = train_hypers['save_log']


    # ###### Loading Testing parameters 
    # test_params = configs['test_params']
    # test_modelload = test_params['model_load_path']
    # test_outputviz = test_params['output_viz']

    

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
    ascii_banner = pyfiglet.figlet_format("Fast GCN !")
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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    ###### Data loading based on the dataset 

    if data_type == 'cora' or data_type == 'citeseer' or data_type == 'pubmed':
       

        citedata = graphdataload.Graph_data(train_datapath,data_type,'Fullsuper')
        citedata.load_data()


        adj = (getattr(citedata, data_type+'_norm_adj'))
        features = (getattr(citedata, data_type+'_features'))
        adj_train = (getattr(citedata, data_type+'_norm_adj_train'))
        train_features = (getattr(citedata, data_type+'_train_features'))
        y_train = (getattr(citedata, data_type+'_y_train'))
        y_test = (getattr(citedata, data_type+'_y_test'))
        train_index = (getattr(citedata, data_type+'_train_index'))
        val_index = (getattr(citedata, data_type+'_val_index'))
        test_index = (getattr(citedata, data_type+'_test_index')) 
        y_val = (getattr(citedata, data_type+'_y_val') )
        classes_num = (getattr(citedata, data_type+'_classnum') )
           

        logging.info("\n[STEP 1]: Processing {} dataset.".format(data_type))
        
        logging.info("| # of nodes : {}".format(adj.shape[0]))
        logging.info("| # of edges : {}".format(adj.sum().sum()/2))  
        logging.info("| # of features : {}".format(features.shape[1]))
        logging.info("| # of clases   : {}".format(classes_num))
        logging.info("| # of train set : {}".format(len(train_index)))
        logging.info("| # of val set   : {}".format(len(val_index)))
        logging.info("| # of test set  : {}".format(len(test_index)))
        logging.info("| # of adj matrix set : {}".format(adj.shape))
        logging.info("| # of norm adj matrix set   : {}".format(adj_train.shape))     
    
    else:
        raise NotImplementedError(data_type)


    #######Data Loading is completed 

    ###Intialization of variables 

    layer_sizes = [128, 128]
    input_dim = features.shape[1]
    train_nums = adj_train.shape[0]    
    nclass = y_train.shape[1]
    nfeats = features.shape[1]
      
    if device.type == 'cuda':

        features = torch.FloatTensor(features).cuda()
        train_features = torch.FloatTensor(train_features).cuda()
        y_train = torch.LongTensor(y_train).cuda().max(1)[1]
        train_ind=np.arange(train_nums)

        test_adj = [adj, adj[test_index, :]]    
        test_feats = features    
        test_labels = y_test    
        test_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).cuda()for cur_adj in test_adj]
        test_labels = torch.LongTensor(test_labels).cuda().max(1)[1]

        val_adj= [adj, adj[val_index, :]]
        val_feats = features
        val_labels= y_val
        val_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).cuda()for cur_adj in val_adj]
        val_labels = torch.LongTensor(val_labels).cuda().max(1)[1]


    if model_type == 'fastgcn':

        logging.info("\n[STEP 2]: Sampler{} definition.".format(model_type))

        sampler = SamplerFastGCN(train_features, adj_train,input_dim,layer_sizes)

        logging.info("\n[STEP 2a]:  Model {} definition.".format(model_type))

    
        model = fastgcn(nfeat=nfeats,
                    nhid=model_hidden,
                    nclass=nclass,
                    dropout=model_droput,
                    sampler=sampler).cuda()

        optimizer = optim.Adam(model.parameters(),lr=train_lr, weight_decay=train_wtdecay)

        logging.info("\n[STEP 2]: Model {} definition.".format(model_type))
        logging.info("Model Architecture Used {}.".format(model_type))   
        logging.info(str(model))
        tot_params = sum([np.prod(p.size()) for p in model.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")
        logging.info(f"Number of epochs: {train_epochs}")

        
        model.to(device)        


        t_total = time.time()
        bad_counter = 0
        best = train_epochs + 1
        best_epoch = 0
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []


        logging.info("\n[STEP 3]: Model {} Training for epochs {}.".format(model_type,train_epochs))        

        for epoch in range(train_epochs):

            to= time.time()
            train_loss,train_acc,val_loss,val_acc = train(train_ind, y_train, train_batch_size,\
                val_feats,val_adj,val_labels, model ,optimizer,valmode= False )

            print('Epoch: {:04d}'.format(epoch+1),
                    'loss_train: {:.4f}'.format(train_loss),
                    'acc_train: {:.4f}'.format(train_acc),
                    'loss_val: {:.4f}'.format(val_loss),
                    'acc_val: {:.4f}'.format(val_acc),
                    'time: {:.4f}s'.format(time.time() - to))

            logging.info("Epoch:{:04d} loss_train:{:.4f} acc_train:{:.4f} loss_val:{:.4f} acc_val:{:.4f} time:{:.4f}s.".format((epoch+1),(train_loss),(train_acc),(val_loss),(val_acc),(time.time()-to)))

            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)

            path = os.path.join(train_modelsave, '{}_{}.pkl'.format(model_type,epoch)) 
            torch.save(model.state_dict(), path)
            if val_loss_history[-1] < best:
                best = val_loss_history[-1]
                best_epoch = epoch
                bad_counter = 0

            else:bad_counter += 1
            
            if bad_counter == train_patience:
                num_epochs = epoch
                break

            for f in glob.glob(os.path.join(train_modelsave,'*.pkl')):
                epoch_nb = int(f.split(os.path.sep)[-1].split('_')[-1].split('.')[0])
                if epoch_nb < best_epoch:
                        os.remove(f)


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


        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        logging.info(f"Total Training Completed :{(time.time() - t_total)}")

        ############################################## Training Completed ##############

        ############################################## Testing Started
        
        loadpath = os.path.join(train_modelsave, '{}_{}.pkl'.format(model_type, best_epoch)) 
        model.load_state_dict(torch.load(loadpath))

        if test_outputviz: 
        
            (save_dir / 'test_fig' if test_outputviz else save_dir).mkdir(parents=True, exist_ok=True)
            testsave_fig = str(save_dir / 'test_fig')

        logging.info("\n[STEP 4]: Testing {} final model.".format(model_type))

        test(test_adj, test_feats, test_labels,model,test_outputviz,testsave_fig,data_type)
     
    else:
        raise NotImplementedError(model_type)


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()





