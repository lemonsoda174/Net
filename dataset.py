import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import read_tiff
import numpy as np
import torchvision
import torchvision.transforms as transforms
import scanpy as sc
from utils import get_data
import os
import glob
from PIL import Image
import pandas as pd 
import scprep as scp
from PIL import ImageFile
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix, minkowski_distance, distance
from collections import defaultdict as dfd
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

random.seed(42)

class ViT_HER2ST(torch.utils.data.Dataset):
    def __init__(self,mode,train=True,gene_list=None,ds=None,sr=False,fold=0,test_size=0.2, te_names = ['A1','B1','C1','D1','E1','F1','G2','H1'], patch_size=112):
        super(ViT_HER2ST, self).__init__()

        #data for the 36 breast cancer sections used in this study
        self.cnt_dir = 'data/her2st/data/ST-cnts' 

        #accompanying histology images
        self.img_dir = 'data/her2st/data/ST-imgs' 

        #list of selected spots for each case, used to subset the raw gene count matrices
        self.pos_dir = 'data/her2st/data/ST-spotfiles'

        #the label with corresponding coordinates of each spot. originally extracted from the annotated HE images
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl' 

        self.r = patch_size//2 #patch size / 2, here, r = 56 --> 112x112 patches
        self.patch_size = patch_size
        #785 genes, filtered from 1000 original - remove gene if they appear in less than 1000 spots
        
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True)) 
        self.gene_list = gene_list
        self.names = os.listdir(self.cnt_dir)
        self.names.sort()
        self.names = [i[:2] for i in self.names]
        self.train = train
        self.sr = sr
        self.mode = mode
        #divide into train/test sets
        
        samples = self.names[0:36]
        testset = ['A1','B1','C1','D1','E1','F1','G2','H1','J1']

        #train set
        tr_names = sorted(list(set(samples) - set(['A1','B1','C1','D1','E1','F1','G2','H1','J1'])))
        print(tr_names, te_names, sep = "\n\n")
        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            self.names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            self.names = te_names
        
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}

        #metadata is a table, including,for each selected spot, coordinates, the number of genes present (per gene type)
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}

        self.label={i:None for i in self.names}

        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names} 
            # self.label={i:m['label'].values for i,m in self.lbl_dict.items()}

            # print(self.lbl_dict)
            # print()
            for i in self.names:
                idx=self.meta_dict[i].index
            # print(idx)
            # print()
                lbl=self.lbl_dict[i]
            # print(lbl)
            # print()
                lbl=lbl.loc[idx,:]['label'].values
            # print(lbl)
            # print()
            # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                self.label[i]=lbl
        elif train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)

                    print(lbl)
                    print("finish")

                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)




        self.gene_set = list(gene_list)

        #gene expression data, normalized and converted to natural log scale
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}

        #get pixel coordinates, rounded down
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}

        #get coordinates (physical distance, in tissue)
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        self.lengths = [len(i) for i in self.meta_dict.values()] #num of spots per sample
        self.cumlen = np.cumsum(self.lengths) #cumulative indexing, considered as global index per sample
        self.id2name = dict(enumerate(self.names)) #index-sample name mapping

        #image augmentation
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        #switch back to (H, W, C) format
        im = im.permute(1,0,2)
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4 


        # output for Histogene:

        # patches: each row is a flattened image patch --> dim = [num_spots, len of row aka 3x112x112]
        # positions: physical coordinates (x,y) of each spot --> dim =  [num_spots, 2]
        # exps: gene expression values of each spot --> dim = [num_spots, total_num_genes aka 785]

        n_patches = len(centers)
        if self.mode == "Histogene":
            patches = torch.zeros((n_patches,patch_dim)) #for histogene only
        elif self.mode == "ST-Net":
            patches = torch.zeros((n_patches, 3, self.patch_size, self.patch_size)) #for stnet only

        exps = torch.Tensor(exps)
        # print(exps.shape)

        for i in range(n_patches):
            center = centers[i]
            x, y = center
            #patch of size 2rx2rx3
            patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:3]
            # print(patch.shape)
            if self.mode == "Histogene":
                patches[i] = patch.flatten() # for histogene only
            elif self.mode == "ST-Net":    
                patch = patch.permute(2, 0, 1) #for stnet only
                patches[i] = patch #.flatten() #for stnet only

        # print(patches.shape, positions.shape, exps.shape, sep = '\n')
        if self.train:
            return patches, positions, exps
        else: 
            return patches, positions, exps, torch.Tensor(centers)
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        # id implies coordinate: (x,y)
        return df

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)

def calcADJ(coord, k=8, distanceType='euclidean', pruneTag='NA'):
    r"""
    Calculate spatial Matrix directly use X/Y coordinates
    """
    spatialMatrix=coord#.cpu().numpy()
    nodes=spatialMatrix.shape[0]
    Adj=torch.zeros((nodes,nodes))
    for i in np.arange(spatialMatrix.shape[0]):
        tmp=spatialMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp,spatialMatrix, distanceType)
        if k == 0:
            k = spatialMatrix.shape[0]-1
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0,res[0][1:k+1]]
        boundary = np.mean(tmpdist)+np.std(tmpdist) #optional
        for j in np.arange(1,k+1):
            # No prune
            if pruneTag == 'NA':
                Adj[i][res[0][j]]=1.0
            elif pruneTag == 'STD':
                if distMat[0,res[0][j]]<=boundary:
                    Adj[i][res[0][j]]=1.0
            # Prune: only use nearest neighbor as exact grid: 6 in cityblock, 8 in euclidean
            elif pruneTag == 'Grid':
                if distMat[0,res[0][j]]<=2.0:
                    Adj[i][res[0][j]]=1.0
    return Adj


class ViT_HER2ST_Hist2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,fold=0,r=4,flatten=True,ori=False,adj=False,prune='Grid',neighs=4,test_size=0.2, te_names = ['A1','B1','C1','D1','E1','F1','G2','H1']):
        super(ViT_HER2ST_Hist2ST, self).__init__()
        
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//r

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        self.names = os.listdir(self.cnt_dir)
        self.names.sort()
        self.names = [i[:2] for i in self.names]
        self.train = train
        self.ori = ori
        self.adj = adj
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = self.names[0:36]
        #test set
        


        #train set
        tr_names = sorted(list(set(samples) - set(te_names)))
        print(tr_names, te_names, sep = "\n\n")
        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            self.names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            self.names = te_names
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        

        self.label={i:None for i in self.names}

        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names} 
            # self.label={i:m['label'].values for i,m in self.lbl_dict.items()}

            # print(self.lbl_dict)
            # print()
            for i in self.names:
                idx=self.meta_dict[i].index
            # print(idx)
            # print()
                lbl=self.lbl_dict[i]
            # print(lbl)
            # print()
                lbl=lbl.loc[idx,:]['label'].values
            # print(lbl)
            # print()
            # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                self.label[i]=lbl
        elif train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)

                    print(lbl)
                    print("finish")

                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)

                    
        self.gene_set = list(gene_list)
        self.exp_dict = {
            i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
            for i,m in self.meta_dict.items()
        }
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) 
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten=flatten
    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1,0,2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label=self.label[ID]
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        return df

    
class HER2ST_Hist2ST(torch.utils.data.Dataset):
    """Patch-level dataset for HER2ST"""
    def __init__(self, train=True, fold=0, r=4, flatten=True, ori=False, adj=False, prune='Grid', neighs=4, test_size=0.2, te_names=['A1','B1','C1','D1','E1','F1','G2','H1']):
        super(HER2ST_Hist2ST, self).__init__()
        
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224 // r

        gene_list = list(np.load('data/her_hvg_cut_1000.npy', allow_pickle=True))
        self.gene_list = gene_list
        self.train = train
        self.ori = ori
        self.adj = adj
        self.flatten = flatten

        all_names = sorted([i[:2] for i in os.listdir(self.cnt_dir)])
        self.names = sorted(list(set(all_names) - set(te_names))) if train else te_names

        print(self.names, te_names, sep="\n\n")

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i: self.get_meta(i) for i in self.names}

        self.label = {i: None for i in self.names}
        self.lbl2id = {
            'invasive cancer': 0, 'breast glands': 1, 'immune infiltrate': 2,
            'cancer in situ': 3, 'connective tissue': 4, 'adipose tissue': 5, 'undetermined': -1
        }

        if not train:
            self.lbl_dict = {i: self.get_lbl(i) for i in self.names}
            for i in self.names:
                idx = self.meta_dict[i].index
                lbl = self.lbl_dict[i].loc[idx, :]['label'].values
                self.label[i] = lbl
        else:
            for i in self.names:
                idx = self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl = self.get_lbl(i).loc[idx, :]['label'].values
                    lbl = torch.Tensor(list(map(lambda x: self.lbl2id[x], lbl)))
                    self.label[i] = lbl
                else:
                    self.label[i] = torch.full((len(idx),), -1)

        self.gene_set = list(gene_list)
        self.exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values))
            for i, m in self.meta_dict.items()
        }

        if self.ori:
            self.ori_dict = {i: m[self.gene_set].values for i, m in self.meta_dict.items()}
            self.counts_dict = {i: m.sum(1) / np.median(m.sum(1)) for i, m in self.ori_dict.items()}

        self.center_dict = {
            i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int)
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}
        self.adj_dict = {
            i: calcADJ(m, neighs, pruneTag=prune)
            for i, m in self.loc_dict.items()
        }

        self.lengths = [len(m) for m in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

    def __len__(self):
        return self.cumlen[-1]  # total number of patches across all slides

    def __getitem__(self, index):
        i = 0
        while index >= self.cumlen[i]:
            i += 1
        patch_idx = index if i == 0 else index - self.cumlen[i - 1]

        ID = self.id2name[i]
        im = self.img_dict[ID]
        im = im.permute(1, 0, 2)

        exp = torch.Tensor(self.exp_dict[ID][patch_idx])
        center = self.center_dict[ID][patch_idx]
        loc = torch.Tensor(self.loc_dict[ID][patch_idx])
        position = torch.LongTensor(self.loc_dict[ID][patch_idx])
        label = self.label[ID][patch_idx]

        if self.ori:
            oris = torch.Tensor(self.ori_dict[ID][patch_idx])
            sfs = torch.Tensor([self.counts_dict[ID][patch_idx]])

        x, y = center
        patch = im[(x - self.r):(x + self.r), (y - self.r):(y + self.r), :]
        if self.flatten:
            patch = patch.flatten()
        else:
            patch = patch.permute(2, 0, 1)

        patch = torch.Tensor(patch)

        data = [patch, position, exp]
        if self.adj:
            data.append(self.adj_dict[ID])
        if self.ori:
            data += [oris, sfs]
        data.append(torch.Tensor(center))
        data.append(torch.Tensor([label]))

        return tuple(data)

    def get_img(self, name):
        path = os.path.join(self.img_dir, name[0], name, os.listdir(os.path.join(self.img_dir, name[0], name))[0])
        return Image.open(path)

    def get_cnt(self, name):
        return pd.read_csv(f'{self.cnt_dir}/{name}.tsv', sep='\t', index_col=0)

    def get_pos(self, name):
        df = pd.read_csv(f'{self.pos_dir}/{name}_selection.tsv', sep='\t')
        df['id'] = [f'{int(round(x))}x{int(round(y))}' for x, y in zip(df['x'], df['y'])]
        return df

    def get_meta(self, name):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        return cnt.join(pos.set_index('id'))

    def get_lbl(self, name):
        df = pd.read_csv(f'{self.lbl_dir}/{name}_labeled_coordinates.tsv', sep='\t')
        df['id'] = [f'{int(round(x))}x{int(round(y))}' for x, y in zip(df['x'], df['y'])]
        df.drop(columns=['pixel_x', 'pixel_y', 'x', 'y'], inplace=True)
        df.set_index('id', inplace=True)
        return df




if __name__ == "__main__":
    dataset = HER2ST(train=True, fold=5)
    sample = dataset[0]