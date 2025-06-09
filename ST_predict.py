import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from models.HisToGene_model import HisToGene
from models.STNet_model import STModel
from models.Hist2ST_model import Hist2ST
from utils import *
from predict import model_predict, get_R, cluster, test
from dataset import ViT_HER2ST, HER2ST, ViT_HER2ST_Hist2ST


fold = 5
tag = '-htg_her2st_785_32_cv'

#normal histogene prediction

mode = input("Choose model to predict [Histogene/ST-Net/Hist2ST]: ")
if mode == "Histogene":
    model = HisToGene.load_from_checkpoint("model_ckpts/histogene_last_train_"+tag+'_'+str(fold)+".ckpt",n_layers=8, n_genes=785, learning_rate=1e-5)
    device = torch.device("cuda")
    dataset = ViT_HER2ST(train=False,sr=False,fold=fold)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=4)

    label=dataset.label[dataset.names[0]]
    adata_pred, adata_gt = model_predict(model, test_loader, attention=False, device = device)
    adata_pred = comp_tsne_km(adata_pred,4)

    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_pred.var_names = g
    sc.pp.scale(adata_pred)
    # print(adata_pred)
    print(adata_pred, adata_gt)

    R=get_R(adata_pred,adata_gt)[0]
    print('Pearson Correlation:',np.nanmean(R))
    clus,ARI=cluster(adata_pred, label)
    print('ARI:',ARI)



    #visualize results
    sc.pl.spatial(adata_pred, img=None, color='kmeans', spot_size=112, save='histogene_kmeans.png')

    sc.pl.spatial(adata_pred, img=None, color='FASN', spot_size=112, color_map='magma', save='histogene_FASN.png')

elif mode == "ST-Net":
    model = STModel.load_from_checkpoint("model_ckpts/stnet_last_train_"+tag+'_'+str(fold)+".ckpt", n_genes=785, learning_rate=1e-5)
    device = torch.device("cuda")
    dataset = HER2ST(train=False,fold=fold)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=4)

    label=dataset.label[dataset.names[0]]
    adata_pred, adata_gt = model_predict(model, test_loader, attention=False, device = device)
    adata_pred = comp_tsne_km(adata_pred,4)

    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_pred.var_names = g
    sc.pp.scale(adata_pred)
    # print(adata_pred)
    print(adata_pred, adata_gt)


    R=get_R(adata_pred,adata_gt)[0]
    print('Pearson Correlation:',np.nanmean(R))
    clus,ARI=cluster(adata_pred, label)
    print('ARI:',ARI)

    #visualize results
    sc.pl.spatial(adata_pred, img=None, color='kmeans', spot_size=112, save='ST-Net_kmeans.png')

    sc.pl.spatial(adata_pred, img=None, color='FASN', spot_size=112, color_map='magma', save='ST-Net_FASN.png')

elif mode == 'Hist2ST':
    values='5-7-2-8-4-16-32-785'
    k,p,d1,d2,d3,h,c,genes=map(lambda x:int(x),values.split('-'))

    model=Hist2ST.load_from_checkpoint("model_ckpts/hist2st_last_train_"+tag+'_'+str(fold)+".ckpt",
        depth1=d1, depth2=d2,depth3=d3,n_genes=genes, 
        kernel_size=k, patch_size=p,
        heads=h, channel=c, dropout=0.2,
        zinb=0.25, nb=False,
        bake=5, lamb=0.5, 
        )
    device = torch.device("cuda")

    dataset = ViT_HER2ST_Hist2ST(
            train=False,fold=fold,flatten=False,
            ori=True,neighs=4,adj=True,prune='Grid',r=4
        )
    test_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
    label=dataset.label[dataset.names[0]]
    
    adata_pred, adata_gt = test(model, test_loader, device = device)

    R=get_R(adata_pred,adata_gt)[0]
    print('Pearson Correlation:',np.nanmean(R))
    clus,ARI=cluster(adata_pred,label)
    print('ARI:',ARI)

    sc.pl.spatial(adata_pred, img=None, color='kmeans', spot_size=112, save='Hist2ST_kmeans.png')
else:
    print('error')