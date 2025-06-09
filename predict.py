import torch
from torch.utils.data import DataLoader
from utils import *
from models.HisToGene_model import HisToGene
import warnings
from dataset import ViT_HER2ST
from tqdm import tqdm
warnings.filterwarnings('ignore')
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score as ari_score

MODEL_PATH = ''


def model_predict(model, test_loader, adata=None, attention=True, device = torch.device('cpu')): 
    model.eval()
    model = model.to(device)
    preds = None
    count = 0
    with torch.no_grad():
        for patch, position, exp, center in tqdm(test_loader):

            patch, position = patch.to(device), position.to(device)
            
            pred = model(patch, position)

            if preds is None:
                preds = pred #previously preds = pred.squeeze(); remove for compatibility w stnet
                ct = center
                gt = exp
            else:
                preds = torch.cat((preds,pred),dim=0)
                ct = torch.cat((ct,center),dim=0)
                gt = torch.cat((gt,exp),dim=0)



    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct

    return adata, adata_gt


def get_R(data1,data2,dim=1,func=pearsonr):
    adata1=data1.X
    adata2=data2.X
    r1,p1=[],[]
    for g in range(data1.shape[dim]):
        if dim==1:
            r,pv=func(adata1[:,g],adata2[:,g])
        elif dim==0:
            r,pv=func(adata1[g,:],adata2[g,:])
        r1.append(r)
        p1.append(pv)
    r1=np.array(r1)
    p1=np.array(p1)
    return r1,p1


def cluster(adata,label):
    idx=label!='undetermined'
    tmp=adata[idx]
    l=label[idx]
    sc.pp.pca(tmp)
    sc.tl.tsne(tmp)
    kmeans = KMeans(n_clusters=len(set(l)), init="k-means++", random_state=0).fit(tmp.obsm['X_pca'])
    p=kmeans.labels_.astype(str)
    lbl=np.full(len(adata),str(len(set(l))))
    lbl[idx]=p
    adata.obs['kmeans']=lbl
    return p,round(ari_score(p,l),3)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # for fold in [5,11,17,26]:
    for fold in range(12):
        # fold=30
        # tag = '-vit_skin_aug'
        # tag = '-cnn_her2st_785_32_cv'
        tag = '-vit_her2st_785_32_cv'
        # tag = '-cnn_skin_134_cv'
        # tag = '-vit_skin_134_cv'
        ds = 'HER2'
        # ds = 'Skin'

        print('Loading model ...')
        # model = STModel.load_from_checkpoint('model/last_train_'+tag+'.ckpt')
        # model = VitModel.load_from_checkpoint('model/last_train_'+tag+'.ckpt')
        # model = STModel.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt") 
        model = HisToGene.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt")
        model = model.to(device)
        # model = torch.nn.DataParallel(model)
        print('Loading data ...')

        # g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        g = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))

        # dataset = SKIN(train=False,ds=ds,fold=fold)
        dataset = ViT_HER2ST(train=False,mt=False,sr=True,fold=fold)
        # dataset = ViT_SKIN(train=False,mt=False,sr=False,fold=fold)
        # dataset = VitDataset(diameter=112,sr=True)

        test_loader = DataLoader(dataset, batch_size=16, num_workers=4)
        print('Making prediction ...')

        adata_pred, adata = model_predict(model, test_loader, attention=False)
        # adata_pred = sr_predict(model,test_loader,attention=True)

        adata_pred.var_names = g
        print('Saving files ...')
        adata_pred = comp_tsne_km(adata_pred,4)
        # adata_pred = comp_umap(adata_pred)
        print(fold)
        print(adata_pred)

        adata_pred.write('processed/test_pred_'+ds+'_'+str(fold)+tag+'.h5ad')
        # adata_pred.write('processed/test_pred_sr_'+ds+'_'+str(fold)+tag+'.h5ad')

        # quit()

#from hist2st
def test(model,test,device='cuda'):
    model=model.to(device)
    model.eval()
    preds=None
    ct=None
    gt=None
    loss=0
    with torch.no_grad():
        for patch, position, exp, adj, *_, center in tqdm(test):
            patch, position, adj = patch.to(device), position.to(device), adj.to(device).squeeze(0)
            pred = model(patch, position, adj)[0]
            preds = pred.squeeze().cpu().numpy()
            ct = center.squeeze().cpu().numpy()
            gt = exp.squeeze().cpu().numpy()
    adata = ad.AnnData(preds)
    adata.obsm['spatial'] = ct
    adata_gt = ad.AnnData(gt)
    adata_gt.obsm['spatial'] = ct
    return adata,adata_gt


if __name__ == '__main__':
    main()

