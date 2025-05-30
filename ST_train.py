from torch.utils.data import DataLoader
from dataset import ViT_HER2ST, HER2ST
from models.HisToGene_model import HisToGene
from models.STNet_model import STModel
import pytorch_lightning as pl

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

fold = 5
tag = '-htg_her2st_785_32_cv'

mode = input("Choose model to train [Histogene/ST-Net]: ")
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
else:
    print("error")