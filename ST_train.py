from torch.utils.data import DataLoader
from dataset import ViT_HER2ST, HER2ST, ViT_HER2ST_Hist2ST
from models.HisToGene_model import HisToGene
from models.STNet_model import STModel
from models.Hist2ST_model import Hist2ST
import pytorch_lightning as pl

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

fold = 5
tag = '-htg_her2st_785_32_cv'

mode = input("Choose model to train [Histogene/ST-Net/Hist2ST]: ")
if mode == "Histogene":
    dataset = ViT_HER2ST(train=True, fold=fold)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
    model = HisToGene(n_layers=8, n_genes=785, learning_rate=1e-5)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=100)
    trainer.fit(model, train_loader)
    trainer.save_checkpoint("model_ckpts/histogene_last_train_"+tag+'_'+str(fold)+".ckpt")
elif mode == "ST-Net":
    dataset = HER2ST(train=True, fold=fold)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
    model = STModel(n_genes=785, learning_rate=1e-5)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=100)
    trainer.fit(model, train_loader)
    trainer.save_checkpoint("model_ckpts/stnet_last_train_"+tag+'_'+str(fold)+".ckpt")
elif mode == "Hist2ST":
    values='5-7-2-8-4-16-32-785'
    k,p,d1,d2,d3,h,c,genes=map(lambda x:int(x),values.split('-'))

    dataset = ViT_HER2ST_Hist2ST(
            train=True,fold=fold,flatten=False,
            ori=True,neighs=4,adj=True,prune='Grid',r=4
        )
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)
    model=Hist2ST(
        depth1=d1, depth2=d2,depth3=d3,n_genes=genes,
        kernel_size=k, patch_size=p,
        heads=h, channel=c, dropout=0.2,
        zinb=0.25, nb=False,
        bake=5, lamb=0.5, 
    )    
    trainer = pl.Trainer(accelerator='gpu', max_epochs=100)
    trainer.fit(model, train_loader)
    trainer.save_checkpoint("model_ckpts/hist2st_last_train_"+tag+'_'+str(fold)+".ckpt")
else:
    print("error")