#!/usr/bin/env python3
"""Nearest-centroid sensitivity probe for DLPFC donor-heldout transfer."""
import argparse,glob,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

class M(nn.Module):
    def __init__(self,n,p,h,l,platform_dim=8):
        super().__init__(); self.platform=nn.Sequential(nn.Linear(platform_dim,32),nn.GELU()); self.encoder=nn.Sequential(nn.Linear(2*n+32,h),nn.LayerNorm(h),nn.GELU(),nn.Linear(h,l))
    def forward(self,x,o,p): return self.encoder(torch.cat([x,o,self.platform(p)],1))

def emb(ms,X,O,P,d):
    out=[]
    with torch.inference_mode():
        for s in range(0,len(X),512):
            xx=torch.from_numpy(X[s:s+512]).to(d); oo=torch.from_numpy(O[s:s+512]).to(d); pp=torch.from_numpy(P[s:s+512]).to(d); out.append(np.mean([m(xx,oo,pp).cpu().numpy() for m in ms],0))
    return np.concatenate(out)

def centroid(A,y,B,classes):
    mu=A.mean(0); sd=A.std(0); sd[sd<1e-6]=1.; A=(A-mu)/sd; B=(B-mu)/sd; C=np.stack([A[y==c].mean(0) for c in classes]); d=((B[:,None,:]-C[None,:,:])**2).mean(2); return np.asarray(classes)[np.argmin(d,1)]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-dir',required=True); ap.add_argument('--checkpoint',action='append',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    fs=sorted(glob.glob(str(Path(a.cache_dir)/'*.npz'))); dm={sid:i for i,g in enumerate([['151507','151508','151509','151510'],['151669','151670','151671','151672'],['151673','151674','151675','151676']]) for sid in g}; rows=[]
    for f in fs:
        z=np.load(f,allow_pickle=False); sid=Path(f).stem; rows.append({'file':sid,'donor':dm[sid],'X':z['E'].astype('float32'),'O':np.repeat(z['panel'].astype('float32'),len(z['E']),0),'P':np.eye(8,dtype='float32')[[0]*len(z['E'])],'y':z['y'].astype(str)})
    genes=np.load(fs[0],allow_pickle=False)['genes'].astype(str); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ms=[]
    for ck in a.checkpoint:
        o=torch.load(ck,map_location='cpu'); c=o['config']; m=M(len(genes),len(c['program_names']),c['hidden'],c['latent'],c.get('platform_dim',8)).to(dev); m.load_state_dict(o['model'],strict=False); m.eval(); ms.append(m)
    for r in rows:r['Z']=emb(ms,r['X'],r['O'],r['P'],dev)
    folds=[]
    for held in [0,1,2]:
        te=[r for r in rows if r['donor']==held]; tr=[r for r in rows if r['donor']!=held]; Xtr=np.concatenate([r['X'] for r in tr]); Ztr=np.concatenate([r['Z'] for r in tr]); ytr=np.concatenate([r['y'] for r in tr]); Xte=np.concatenate([r['X'] for r in te]); Zte=np.concatenate([r['Z'] for r in te]); yte=np.concatenate([r['y'] for r in te]); classes=sorted(set(ytr)&set(yte)); keep=np.isin(ytr,classes); keepte=np.isin(yte,classes); Xtr=Xtr[keep]; Ztr=Ztr[keep]; ytr=ytr[keep]; Xte=Xte[keepte]; Zte=Zte[keepte]; yte=yte[keepte]; pca=PCA(n_components=32,svd_solver='randomized',iterated_power=2,random_state=170+held).fit(Xtr); Ptr=pca.transform(Xtr); Pte=pca.transform(Xte); aa=np.array([classes.index(x) for x in ytr]); bb=np.array([classes.index(x) for x in yte]); preds={'model':centroid(Ztr,aa,Zte,list(range(len(classes)))),'raw':centroid(Xtr,aa,Xte,list(range(len(classes)))),'pca':centroid(Ptr,aa,Pte,list(range(len(classes))))}; row={'heldout_donor':held,'test_files':[r['file'] for r in te],'n_train':int(len(aa)),'n_test':int(len(bb)),'classes':classes}; row.update({name:{'macro_f1':float(f1_score(bb,p,average='macro',labels=np.arange(len(classes)),zero_division=0)),'accuracy':float(np.mean(p==bb))} for name,p in preds.items()}); folds.append(row)
    out={'stage':'spatiallibd_centroid','source':'spatialLIBD Human DLPFC Visium','task':'spot-level cortical layer','holdout':'leave-one-donor-out','probe':'nearest class centroid with train-only standardization; PCA train-only','folds':folds,'note':'Frozen label_heldout checkpoints; no target-donor adaptation.'}; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
