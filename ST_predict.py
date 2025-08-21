import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from models.HisToGene_model import HisToGene
from models.STNet_model import STModel
from models.Hist2ST_model import Hist2ST
from models.TCGN_model import TCGNModel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import *
from predict import model_predict, get_R, cluster, test, get_MAE, get_MSE
from dataset import ViT_HER2ST, HER2ST, ViT_HER2ST_Hist2ST
import random

#

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)

# Set the desired seed
seed_value = 102
set_seed(seed_value)

fold = 5
tag = '-htg_her2st_785_32_cv'
# te_names = ['C1']
te_names = ['A1','B1','C1','D1','E1','F1','G2','H1']


img = mpimg.imread("/mnt/disk1/nhdang/spatial_transcriptomics/Net/plot/" + te_names[0] + "_new.png")
# te_names = ['H1']
# img = mpimg.imread("/mnt/disk1/nhdang/spatial_transcriptomics/Net/data/her2st/data/ST-imgs/C/C1/5714_HE_BT_C1.jpg")

patch_level = False
#normal histogene prediction


mode = "Histogene"
if mode == "Histogene":
    model = HisToGene.load_from_checkpoint("model_ckpts/histogene_last_train_"+tag+'_'+str(fold)+"_slide_level"+".ckpt",n_layers=8, n_genes=785, learning_rate=1e-5, patch_size=112, patch_level = False)
    device = torch.device("cuda")
    dataset = ViT_HER2ST(train=False, fold=fold, patch_size=112, te_names = te_names, mode = mode)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=4)
   
    label = None
    print(len(dataset))
    #iterate over labels of test set

    for i in range(len(dataset)):
        if label is None:
            label=dataset.label[dataset.names[i]]
            # print(label.shape)
        else:
            temp=dataset.label[dataset.names[i]]
            label=np.concatenate((label,temp))
            # print(temp.shape)
    # print(label)
    # print(label.shape)
    # print(dataset.names)
    print("check bef")
    adata_pred, adata_gt = model_predict(model, test_loader, model_type = mode, attention=False, device = device)
    print("check")
    adata_pred = comp_tsne_km(adata_pred,4)


    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_pred.var_names = g
    sc.pp.scale(adata_pred)
    # print(adata_pred)
    print(adata_pred, adata_gt)

    R=get_R(adata_pred,adata_gt)[0]
    print(R.shape)
    MSE = get_MSE(adata_pred, adata_gt)
    MAE = get_MAE(adata_pred, adata_gt)
    print('MSE:', np.nanmean(MSE))
    print('MAE:', np.nanmean(MAE))

    print('Pearson Correlation:',np.nanmean(R))
    clus,ARI=cluster(adata_pred, label)
    print('ARI:',ARI)
    

    


    # visualize results

    # sc.pl.spatial(adata_pred, img=img, color='kmeans', spot_size=112, frameon=False,
    # legend_loc=None,title=None,
    # show=False)

    # ax = plt.gca()
    # ax.set_title("")  # Remove title

    # plt.savefig(f"figures/kmeans/histogene_kmeans_{te_names[0]}_new.png", dpi=300, bbox_inches='tight', transparent=True)
    # plt.close()

    # sc.pl.spatial(adata_pred, img=img, color='FASN', spot_size=112, frameon=False,
    # legend_loc=None, title=None,
    # show=False,color_map='magma')

    # ax = plt.gca()
    # ax.set_title("")  # Remove title

    # plt.savefig(f"figures/FASN/histogene_FASN_{te_names[0]}_new.png", dpi=300, bbox_inches='tight', transparent=True)
    # plt.close()




elif mode == "ST-Net":
    model = STModel.load_from_checkpoint("model_ckpts/stnet_last_train_"+tag+'_'+str(fold)+"_slide_level"+".ckpt", n_genes=785, learning_rate=1e-5)
    device = torch.device("cuda")
    dataset = ViT_HER2ST(train=False, fold=fold, patch_size=112, te_names = te_names, mode = mode)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=4)

    label = None
    print(len(dataset))
    #iterate over labels of test set

    for i in range(len(dataset)):
        if label is None:
            label=dataset.label[dataset.names[i]]
            # print(label.shape)
        else:
            temp=dataset.label[dataset.names[i]]
            label=np.concatenate((label,temp))


    # label=dataset.label[dataset.names[0]]
    adata_pred, adata_gt = model_predict(model, test_loader, model_type = mode, attention=False, device = device)
    adata_pred = comp_tsne_km(adata_pred,4)

    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_pred.var_names = g
    sc.pp.scale(adata_pred)
    # print(adata_pred)
    print(adata_pred, adata_gt)


    R=get_R(adata_pred,adata_gt)[0]

    MSE = get_MSE(adata_pred, adata_gt)
    MAE = get_MAE(adata_pred, adata_gt)
    print('MSE:', np.nanmean(MSE))
    print('MAE:', np.nanmean(MAE))

    print('Pearson Correlation:',np.nanmean(R))
    clus,ARI=cluster(adata_pred, label)
    print('ARI:',ARI)



    #visualize results
    sc.pl.spatial(adata_pred, img=img, color='kmeans', spot_size=112, frameon=False,
    legend_loc=None,title=None,
    show=False)

    ax = plt.gca()
    ax.set_title("")  # Remove title

    plt.savefig(f"figures/kmeans/ST-Net_kmeans_{te_names[0]}_new.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    sc.pl.spatial(adata_pred, img=img, color='FASN', spot_size=112, frameon=False,
    legend_loc=None, title=None,
    show=False,color_map='magma')

    ax = plt.gca()
    ax.set_title("")  # Remove title

    plt.savefig(f"figures/FASN/ST-Net_FASN_{te_names[0]}_new.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    

elif mode == 'Hist2ST':
    values='5-7-2-8-4-16-32-785'
    k,p,d1,d2,d3,h,c,genes=map(lambda x:int(x),values.split('-'))

    model=Hist2ST.load_from_checkpoint("model_ckpts/hist2st_last_train_"+tag+'_'+str(fold)+"_slide_level"+".ckpt",
        depth1=d1, depth2=d2,depth3=d3,n_genes=genes, 
        kernel_size=k, patch_size=p,
        heads=h, channel=c, dropout=0.2,
        zinb=0.25, nb=False,
        bake=5, lamb=0.5, patch_level = False
        )
    device = torch.device("cuda")

    dataset = ViT_HER2ST_Hist2ST(
            train=False,fold=fold,flatten=False,
            ori=True,neighs=4,adj=True,prune='Grid',r=4, te_names = te_names
        )
    test_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)


    label = None
    print(len(dataset))
    #iterate over labels of test set
    if not patch_level:
        for i in range(len(dataset)):
            if label is None:
                label=dataset.label[dataset.names[i]]
                # print(label.shape)
            else:
                temp=dataset.label[dataset.names[i]]
                label=np.concatenate((label,temp))
    adata_pred, adata_gt = test(model, test_loader, device = device)
    adata_pred = comp_tsne_km(adata_pred,4)


    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_pred.var_names = g
    sc.pp.scale(adata_pred)
    # print(adata_pred)
    print(adata_pred, adata_gt)


    R=get_R(adata_pred,adata_gt)[0]
    MSE = get_MSE(adata_pred, adata_gt)
    MAE = get_MAE(adata_pred, adata_gt)
    print('Pearson Correlation:',np.nanmean(R))
    print('MSE:', np.nanmean(MSE))
    print('MAE:', np.nanmean(MAE))
    clus,ARI=cluster(adata_pred,label)
    print('ARI:',ARI)

    
    # visualize

    # sc.pl.spatial(adata_pred, img=img, color='kmeans', spot_size=112, frameon=False, legend_loc=None,title=None,show=False)
    # ax = plt.gca()
    # ax.set_title("")
    # plt.savefig(f"figures/kmeans/Hist2ST_kmeans_{te_names[0]}.png", dpi=300, bbox_inches='tight', transparent=True)
    # plt.close()

    # sc.pl.spatial(adata_pred, img=img, color='FASN', spot_size=112, frameon=False, legend_loc=None, title=None, show=False,color_map='magma')
    # ax = plt.gca()
    # ax.set_title("")
    # plt.savefig(f"figures/FASN/Hist2ST_FASN_{te_names[0]}.png", dpi = 300, bbox_inches='tight', transparent=True)
    # plt.close()

elif mode == "TCGN":
    model = TCGNModel.load_from_checkpoint("model_ckpts/tcgn_last_fold5_patch_level.ckpt", n_genes=785, learning_rate=1e-5)
    device = torch.device("cuda")
    dataset = HER2ST(train=False,fold=fold, patch_level = False, te_names = te_names)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=4)

    label = None
    print(len(dataset))
    #iterate over labels of test set
    if not patch_level:
        for name in dataset.names:  # iterate directly over slide names
            temp = dataset.label[name]
            if label is None:
                label = temp
            else:
                label = np.concatenate((label, temp))

    adata_pred, adata_gt = model_predict(model, test_loader, attention=False, device = device)
    adata_pred = comp_tsne_km(adata_pred,4)

    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_pred.var_names = g
    sc.pp.scale(adata_pred)
    # print(adata_pred)
    print(adata_pred, adata_gt)


    R=get_R(adata_pred,adata_gt)[0]

    # MSE = get_MSE(adata_pred, adata_gt)
    # MAE = get_MAE(adata_pred, adata_gt)
    # print('MSE:', np.nanmean(MSE))
    # print('MAE:', np.nanmean(MAE))

    print('Pearson Correlation:',np.nanmean(R))
    clus,ARI=cluster(adata_pred, label)
    print('ARI:',ARI)

    #visualize results
    sc.pl.spatial(adata_pred, img=img, color='kmeans', spot_size=112, frameon=False, legend_loc=None,title=None,show=False)
    ax = plt.gca()
    ax.set_title("")
    plt.savefig(f"figures/kmeans/TCGN_kmeans_{te_names[0]}.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    sc.pl.spatial(adata_pred, img=img, color='FASN', spot_size=112, frameon=False, legend_loc=None, title=None, show=False,color_map='magma')
    ax = plt.gca()
    ax.set_title("")
    plt.savefig(f"figures/FASN/TCGN_FASN_{te_names[0]}.png", dpi = 300, bbox_inches='tight', transparent=True)
    plt.close()

else:
    print('error')