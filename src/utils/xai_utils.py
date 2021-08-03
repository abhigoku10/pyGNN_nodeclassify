import torch
import numpy as np 
from tqdm import tqdm

from ..models.graph_lime import GraphLIME
from ..models.xai_explainers import LIME, Greedy, Random
from torch_geometric.nn import GNNExplainer

def modify_trainmask(num_nodes,train_mask,val_mask,test_mask):

    ####----- To modify the train mask data 
    
    num_train = int(num_nodes * 0.8)

    node_indices = np.random.choice(num_nodes, size=num_train, replace=False)
    new_train_mask = torch.zeros_like(train_mask)
    new_train_mask[node_indices] = 1
    new_train_mask = new_train_mask > 0

    new_val_mask = torch.zeros_like(val_mask)
    new_val_mask = new_val_mask > 0

    new_test_mask = ~(new_train_mask + new_val_mask)

    train_mask = new_train_mask
    val_mask = new_val_mask
    test_mask = new_test_mask

    # val_mask = data.val_mask
    # test_mask = data.test_mask
    # new_train_mask = ~(val_mask + test_mask)

    # data.train_mask = new_train_mask
    
    return train_mask , val_mask, test_mask

def  add_noise_features(x,num_nodes, num_noise):

    if not num_noise:
        return x

    # num_nodes = data.x.size(0)

    noise_feat = torch.randn((num_nodes, num_noise))
    noise_feat = noise_feat - noise_feat.mean(1, keepdim=True)
    noise_feat =noise_feat.cuda()

    x = torch.cat([x, noise_feat], dim=1)
    
    return x

def extract_test_nodes(test_mask, num_samples):
    test_indices = test_mask.cpu().numpy().nonzero()[0]
    node_indices = np.random.choice(test_indices, num_samples).tolist()

    return node_indices

def find_noise_feats_by_GraphLIME(model ,x,edge_idx, test_mask,hop,rho,test_samples,input_dim,K):

    explainer = GraphLIME(model, hop=hop, rho=rho)

    node_indices = extract_test_nodes(test_mask,test_samples)

    num_noise_feats = []

    for node_idx in tqdm(node_indices, desc='explain node', leave=False):
        coefs = explainer.explain_node(node_idx, x, edge_idx)

        # print("The node_idx is {} coefficients are {}".format(node_idx,coefs))

        feat_indices = coefs.argsort()[-K:]
        feat_indices = [idx for idx in feat_indices if coefs[idx] > 0.0]

        num_noise_feat = sum(idx >= input_dim for idx in feat_indices)
        num_noise_feats.append(num_noise_feat)

    return num_noise_feats



def find_noise_feats_by_GNNExplainer(model, x,edge_index,test_mask,masks_epochs,masks_lr,hop,test_samples,K, masks_threshold,ip_dim):
    explainer = GNNExplainer(model, epochs=masks_epochs, lr=masks_lr, num_hops=hop, log=False)

    node_indices =extract_test_nodes(test_mask,test_samples)

    num_noise_feats = []
    for node_idx in tqdm(node_indices, desc='explain node', leave=False):
        node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
        node_feat_mask = node_feat_mask.detach().cpu().numpy()

        feat_indices = node_feat_mask.argsort()[-K:]
        feat_indices = [idx for idx in feat_indices if node_feat_mask[idx] > masks_threshold]

        num_noise_feat = sum(idx >= ip_dim for idx in feat_indices)
        num_noise_feats.append(num_noise_feat)
    
    return num_noise_feats


def find_noise_feats_by_LIME(model,x,edge_index,test_mask,lime_samples,test_samples, input_dim,K):
    explainer = LIME(model, lime_samples)

    node_indices = extract_test_nodes(test_mask,test_samples)

    num_noise_feats = []
    for node_idx in tqdm(node_indices, desc='explain node', leave=False):
        coefs = explainer.explain_node(node_idx, x, edge_index)
        coefs = np.abs(coefs)

        # print("The node_idx is {} coefficients are {}".format(node_idx,coefs))

        feat_indices = coefs.argsort()[-K:]

        num_noise_feat = sum(idx >= input_dim for idx in feat_indices)
        num_noise_feats.append(num_noise_feat)

    return num_noise_feats


def find_noise_feats_by_greedy(model,x,edge_index,test_samples,test_mask, greedy_threshold,ip_dim,K):
    explainer = Greedy(model)

    node_indices = extract_test_nodes(test_mask,test_samples)

    delta_probas = explainer.explain_node(node_indices, x, edge_index)  # (#test_smaples, #feats)
    feat_indices = delta_probas.argsort(axis=-1)[:, -K:]  # (#test_smaples, K)

    num_noise_feats = []
    for node_proba, node_feat_indices in zip(delta_probas, feat_indices):
        node_feat_indices = [feat_idx for feat_idx in node_feat_indices if node_proba[feat_idx] > greedy_threshold]
        num_noise_feat = sum(feat_idx >= ip_dim for feat_idx in node_feat_indices)
        num_noise_feats.append(num_noise_feat)

    return num_noise_feats


def find_noise_feats_by_random(features,test_samples, ip_dim,K):
    num_feats = features.size(1)
    explainer = Random(num_feats, K)

    num_noise_feats = []
    for node_idx in tqdm(range(test_samples), desc='explain node', leave=False):
        feat_indices = explainer.explain_node()
        noise_feat = (feat_indices >= ip_dim).sum()
        num_noise_feats.append(noise_feat)

    return num_noise_feats

