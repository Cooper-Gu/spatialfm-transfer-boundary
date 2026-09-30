#!/usr/bin/env python3
"""Independent SpatialFusion LUAD region-label block-heldout probe."""
import argparse,json
from pathlib import Path
import numpy as np, torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

class MaskAware(nn.Module):
    def __init__(self,n,p,h,l,platform_dim=8):
        super().__init__(); self.platform=nn.Sequential(nn.Linear(platform_dim,32),nn.GELU()); self.encoder=nn.Sequential(nn.Linear(2*n+32,h),nn.LayerNorm(h),nn.GELU(),nn.Linear(h,l)); self.local_head=nn.Linear(l,p)
    def forward(self,x,o,p):
        z=self.encoder(torch.cat([x,o,self.platform(p)],1)); return z,self.local_head(z)

def embed(models,X,O,P,dev):
    out=[]
    with torch.inference_mode():
        for st in range(0,len(X),256):
            xx=torch.from_numpy(X[st:st+256]).to(dev); oo=torch.from_numpy(O[st:st+256]).to(dev); pp=torch.from_numpy(P[st:st+256]).to(dev)
            out.append(np.mean([m(xx,oo,pp)[0].cpu().numpy() for m in models],axis=0))
    return np.concatenate(out,0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--npz',required=True); ap.add_argument('--checkpoint',action='append',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,default=20262026); ap.add_argument('--bootstrap',type=int,default=100); a=ap.parse_args(); rng=np.random.default_rng(a.seed)
    z=np.load(a.npz,allow_pickle=False); X=z['E'].astype('float32'); O=np.repeat(z['panel'].astype('float32'),len(X),axis=0); P=np.zeros((len(X),8),dtype='float32'); P[:,0]=1.; coord=z['coord'].astype('float32'); labels=z['region_label'].astype(str); keep=~np.isin(labels,['Unassigned','nan','None','__missing__']); X=X[keep]; O=O[keep]; P=P[keep]; coord=coord[keep]; labels=labels[keep]
    # 4x4 spatial quantile grid; checkerboard folds keep contiguous blocks apart.
    ix=np.clip(np.searchsorted(np.quantile(coord[:,0],[.25,.5,.75]),coord[:,0]),0,3); iy=np.clip(np.searchsorted(np.quantile(coord[:,1],[.25,.5,.75]),coord[:,1]),0,3); fold=(ix+iy)%2
    genes=z['genes'].astype(str); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); models=[]
    for ck in a.checkpoint:
        o=torch.load(ck,map_location='cpu'); c=o['config']; m=MaskAware(len(genes),len(c['program_names']),c['hidden'],c['latent'],c.get('platform_dim',8)).to(dev); m.load_state_dict(o['model'],strict=False); m.eval(); models.append(m)
    Z=embed(models,X,O,P,dev); results=[]
    for fi in [0,1]:
        tr=fold!=fi; te=fold==fi; classes=sorted(set(labels[tr].tolist())&set(labels[te].tolist())); tr&=np.isin(labels,classes); te&=np.isin(labels,classes); cmap={c:i for i,c in enumerate(classes)}; ytr=np.array([cmap[x] for x in labels[tr]]); yte=np.array([cmap[x] for x in labels[te]])
        pca=PCA(n_components=min(32,int(tr.sum())-1,X.shape[1]),random_state=a.seed).fit(X[tr]); methods={'maskaware':(Z[tr],Z[te]),'raw':(X[tr],X[te]),'pca':(pca.transform(X[tr]),pca.transform(X[te]))}; mr={'fold':fi,'n_train':int(tr.sum()),'n_test':int(te.sum()),'classes':classes}
        for name,(A,B) in methods.items():
            clf=LogisticRegression(max_iter=250,C=1.0,class_weight='balanced',random_state=a.seed,n_jobs=2); clf.fit(A,ytr); pred=clf.predict(B); boots=[]
            for _ in range(a.bootstrap):
                bi=rng.integers(0,len(yte),size=len(yte)); boots.append(float(f1_score(yte[bi],pred[bi],average='macro',labels=np.arange(len(classes)))))
            mr[name]={'macro_f1':float(f1_score(yte,pred,average='macro')),'accuracy':float(np.mean(pred==yte)),'bootstrap95':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],'per_class_f1':{c:float(v) for c,v in zip(classes,f1_score(yte,pred,average=None,labels=np.arange(len(classes))))}}
        results.append(mr)
    out={'stage':'luad_region','label':'region_annotation','source':'SpatialFusion LUAD independent cache','n_total':int(len(X)),'label_counts':{str(k):int(v) for k,v in zip(*np.unique(labels,return_counts=True))},'checkpoints':a.checkpoint,'folds':results,'note':'Expression cache is not present in public_regions_union6k_pilot manifest; Unassigned regions excluded; checkerboard spatial block-heldout probe.'}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
