
import os 
import torch 
import sys
sys.path.append(os.getcwd())
import time


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D



# visdom display module
from visdom import Visdom
import numpy as np 
import pandas as pd



# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib



# Random state.
RS = 20191101

# References : 
# https://github.com/taishan1994/pytorch_gat/blob/master/main.py


# t-SNE Dimensionality reduction
def t_SNE(output, test_labels,dimention,save_path):
    '''
    https://github.com/HanGuangXin/Result-Visualization-of-Graph-Convolutional-Networks-in-PyTorch
    
    https://github.com/thanhtrunghuynh93/pygcn/blob/master/gcn.ipynb

    '''
    # output: data to be reduced in dimensionality
    # dimention: Reduced dimension
    tsne = TSNE(n_components=dimention, init='pca', random_state=0)
    result = tsne.fit_transform(output)


    if len(test_labels.shape)>=2:
        if torch.is_tensor(test_labels):
            test_labels = test_labels.cpu().numpy()
        he = np.argmax(test_labels,axis=1)
        plt.scatter(result[:,0], result[:,1], marker='o', s=5, c=he)
    else:
        plt.scatter(result[:,0], result[:,1], marker='o', s=5, c=test_labels)
    pathfile = os.path.join(save_path,"tsne_plt.png")
    plt.savefig(pathfile)
    plt.close()
    return result



def pca_tsne(output,labels,save_path):
    '''
    https://github.com/thanhtrunghuynh93/pygcn

    '''

    feat_cols = [ 'dim'+str(i) for i in range(output.shape[1])]
    if torch.is_tensor(output):
        df = pd.DataFrame(output.detach().cpu().numpy(), columns=feat_cols)
    # else:
    #     df = pd.DataFrame(output, columns=feat_cols)
    #     if torch.is_tensor(labels)== False and len(labels.shape)==1:
    #         df['label'] = labels

         

    if len(labels.shape)>=2:
        if torch.is_tensor(labels):
            h2 = labels.cpu().numpy()
        num_class = h2.shape[1]
        df['label'] = np.argmax(h2,axis=1)
    elif torch.is_tensor(labels):
        num_class= output.shape[1]
        df['label'] = labels.cpu().numpy()
    
    print('Size of the dataframe: {}'.format(df.shape))

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    
    plt.figure(figsize=(16,10))
    ### need to make the plot class independent 
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="label",
        palette=sns.color_palette("hls", num_class),
        data=df.loc[:,:],
        legend="full",
        alpha=0.3
    )
    # plt.show()
    pathfile = os.path.join(save_path,"2D_pca_one_two.png")
    plt.savefig(pathfile)
    plt.close()
    
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=df.loc[:,:]["pca-one"], 
        ys=df.loc[:,:]["pca-two"], 
        zs=df.loc[:,:]["pca-three"], 
        c=df.loc[:,:]["label"], 
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    # plt.show()
    pathfile = os.path.join(save_path,"3D_pca_one_two.png")
    plt.savefig(pathfile)
    plt.close()

    
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=250)
    tsne_results = tsne.fit_transform(df[feat_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette=sns.color_palette("hls", num_class),
        data=df,
        legend="full",
        alpha=0.3
        )
    # plt.show()
    pathfile = os.path.join(save_path,"TSNE.png")
    plt.savefig(pathfile)
    plt.close()


### TSNE vis with legend 
def tsne_legend(x, y, labels, title, save_path,name='tsne'):
    '''
    https://github.com/zhulf0804/GCN.PyTorch
    
    t-sne visualization.
    :param x: (n, 2)
    :param y: (n, )
    :param labels: (class_num, )
    :param title: used for plt.title and saved name
    :param name: used for saved filename
    :return:
    '''
    x = TSNE(random_state=RS).fit_transform(x)
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", len(labels)))
    ax = plt.subplot(aspect='equal')
    ax.scatter(x[:,0], x[:,1], lw=0, s=40,c=palette[y.astype(np.int)])
    ax.axis('on')
    # ax.axis('tight')

    # add the labels for each digit.
    txts = []
    for i in range(len(labels)):
        # Position of each label.
        xtext, ytext = np.median(x[y == i, :], axis=0)
        txt = ax.text(xtext, ytext, labels[i], fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.title(title)
    # plt.savefig(os.path.join(saved_dir, "%s_%s.png" %(name, title)), dpi=120)

    pathfile = os.path.join(save_path, "%s_%s.png" %(name, title))
    plt.savefig(pathfile,dpi=120)
    plt.close()
    

# Visualization with visdom
def viz_visdom(vis, result, labels,title):
    '''
    https://github.com/HanGuangXin/Result-Visualization-of-Graph-Convolutional-Networks-in-PyTorch
    '''
    # vis: Visdom object
    # result: The data to be displayed, here is the output of the t_SNE() function
    # label: The label of the data to be displayed
    # title: title
    vis . scatter (
        X = result,
        Y  =  labels + 1 ,            # Change the minimum value of label from 0 to 1, and label cannot be 0 when displayed
       opts=dict(markersize=4,title=title),
    )


def scatter_mnist(x, colors):

    '''
    Ref: https://github.com/oreillymedia/t-SNE-tutorial
    https://github.com/shivanichander/tSNE
    '''
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def viz_mnist_2d(dataloader,batch_size):

    examples = enumerate(dataloader)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data.shape
    fig = plt.figure()
    num_samples = 12 
    assert num_samples <= batch_size
    grid_row = 3
    grid_col = num_samples/grid_row
    for i in range(num_samples):
        plt.subplot(grid_row,grid_col,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()



def viz_mnist_tsne(dataloader):
    examples = enumerate(dataloader)
    batch_idx, (example_data, example_targets) = next(examples)
    n_samples, n_features,nx,ny = example_data.shape
    example_data = example_data.reshape((n_samples,nx*ny))
    
    print('Computing MNIST t-SNE embedding')
    # We first reorder the data points according to the handwritten numbers.
    X = np.vstack([example_data[example_targets==i]
                for i in range(10)])
    y = np.hstack([example_targets[example_targets==i]
                for i in range(10)])

    ##Implementationv1    
    # result = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=300).fit_transform(X)
    # plot_embed_mnist(result, y, 't-SNE embedding of the digits (time %.2fs)')
    digits_proj = TSNE(random_state=20150101).fit_transform(X)
    scatter_mnist(digits_proj, y)
    plt.savefig('digits_tsne-generated.png', dpi=120)



def plot_train_val_acc(num_epochs,train_acc_history,val_acc_history,save_path):
    '''
    https://github.com/taishan1994/pytorch_gat
    '''
    plt.plot(num_epochs, train_acc_history, 'b--')
    plt.plot(num_epochs, val_acc_history, 'r-')
    plt.title('Training and validation Acc ')
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.legend(['train_acc','val_acc'])
    pathfile = os.path.join(save_path,"acc.png")
    plt.savefig(pathfile)
    plt.close()



def plot_train_val_loss(num_epochs,train_loss_history,val_loss_history,save_path):
    '''
    https://github.com/taishan1994/pytorch_gat
    '''
    plt.plot(num_epochs, train_loss_history, 'b--')
    plt.plot(num_epochs, val_loss_history, 'r-')
    plt.title('Training and validation Loss ')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["train_loss", 'val_loss'])
    pathfile = os.path.join(save_path,"loss.png")
    plt.savefig(pathfile)
    plt.close()

def xai_plot_dist(noise_feats, label=None, ymax=1.0, color=None, title=None, save_path=None):
    sns.set_style('darkgrid')
    ax = sns.distplot(noise_feats, hist=False, kde=True, kde_kws={'label': label}, color=color)
    plt.xlim(-3, 11)
    plt.ylim(ymin=0.0, ymax=ymax)

    if title:
        plt.title(title)
        
    if save_path:
        plt.savefig(save_path)

    return ax
    
