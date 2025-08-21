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
from predict import model_predict, get_R, cluster, test, get_MSE, get_MAE
from dataset import ViT_HER2ST, ViT_HER2ST_Hist2ST


te_name = [['A1'],['B1'],['C1'],['D1'],['E1'],['F1'],['G2'],['H1']]

fold = 5
tag = '-htg_her2st_785_32_cv'
patch_level = False

for te_names in te_name:
    img = mpimg.imread(f"/mnt/disk1/nhdang/spatial_transcriptomics/Net-pricai01/data/imgs_whitened/{te_names[0]}_new.png")

    model = HisToGene.load_from_checkpoint("model_ckpts/histogene_last_train_"+tag+'_'+str(fold)+"_slide_level"+".ckpt",n_layers=8, n_genes=785, learning_rate=1e-5, patch_size=112, patch_level = False)
    device = torch.device("cuda")
    dataset = ViT_HER2ST(train=False, fold=fold, patch_size=112, te_names = te_names, mode = "Histogene")
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
    adata_pred, adata_gt = model_predict(model, test_loader, model_type = "Histogene", attention=False, device = device)
    print("check")
    adata_pred = comp_tsne_km(adata_pred,4)
    adata_gt = comp_tsne_km(adata_gt,4)

    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_pred.var_names = g
    adata_gt.var_names = g
    sc.pp.scale(adata_pred)
    # print(adata_pred)
    print(adata_pred, adata_gt)

    R=get_R(adata_pred,adata_gt)[0]
    # MSE = get_MSE(adata_pred, adata_gt)
    # MAE = get_MAE(adata_pred, adata_gt)
    # print('MSE:', np.nanmean(MSE))
    # print('MAE:', np.nanmean(MAE))

    print('Pearson Correlation:',np.nanmean(R))

    clus,ARI=cluster(adata_gt, label)
    print('ARI:',ARI)





    # visualize results

    sc.pl.spatial(adata_gt, img=img, color='kmeans', spot_size=112, frameon=False,
    legend_loc=None,title=None,
    show=False)

    ax = plt.gca()
    ax.set_title("")  # Remove title

    plt.savefig(f"figures/kmeans/GT_kmeans_{te_names[0]}_new.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    # sc.pl.spatial(adata_pred, img=img, color='FASN', spot_size=112, frameon=False,
    # legend_loc=None, title=None,
    # show=False,color_map='magma')

    # ax = plt.gca()
    # ax.set_title("")  # Remove title

    # plt.savefig(f"figures/FASN/histogene_FASN_{te_names[0]}_new.png", dpi=300, bbox_inches='tight', transparent=True)
    # plt.close()