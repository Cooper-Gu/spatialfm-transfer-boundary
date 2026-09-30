#!/usr/bin/env python3
"""Independent Xenium/Novae coarse cell-label probe for frozen mask-aware models."""
import argparse, glob, json, os
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
            zs=[m(xx,oo,pp)[0].cpu().numpy() for m in models]; out.append(np.mean(zs,axis=0))
    return np.concatenate(out,0)

def onehot_platform(v):
    v=np.asarray(v).astype(int); out=np.zeros((len(v),8),dtype=np.float32); out[:,7]=1
    for i,x in enumerate(v):
        if 0<=x<=6: out[i]=0; out[i,x]=1
    return out

def run_split(rows,train_names,test_names,models,genes,seed,max_per_file,dev,bootstrap=200):
    rng=np.random.default_rng(seed); tr=[]; te=[]; overlap=[]
    for r in rows:
        z=np.load(r,allow_pickle=False); n=len(z['E']); jj=rng.choice(n,size=min(max_per_file,n),replace=False)
        X=z['E'][jj].astype('float32'); panel=z['panel'].astype('float32'); O=np.repeat(panel,len(jj),axis=0); y=z['tissue_label'][jj].astype(str)
        P=onehot_platform(np.zeros(len(jj),dtype=int))
        rec={'name':os.path.basename(r),'X':X,'O':O,'P':P,'y':y,'genes':z['genes'].astype(str)}
        (tr if os.path.basename(r) in train_names else te).append(rec)
        overlap.append(int(np.sum(panel>0)))
    train=np.concatenate([x['X'] for x in tr]); test=np.concatenate([x['X'] for x in te]);
    ytr=np.concatenate([x['y'] for x in tr]); yte=np.concatenate([x['y'] for x in te]); test_file_id=np.concatenate([np.full(len(x['y']),i,dtype=np.int32) for i,x in enumerate(te)])
    classes=sorted(set(ytr.tolist()) & set(yte.tolist())); keeptr=np.isin(ytr,classes); keepte=np.isin(yte,classes); train=train[keeptr]; ytr=ytr[keeptr]; test=test[keepte]; yte=yte[keepte]; test_file_id=test_file_id[keepte]
    # Panels are binary measured-gene masks. All caches share the same union gene order.
    Otr=np.concatenate([x['O'] for x in tr])[keeptr]; Ote=np.concatenate([x['O'] for x in te])[keepte]; Ptr=np.concatenate([x['P'] for x in tr])[keeptr]; Pte=np.concatenate([x['P'] for x in te])[keepte]
    Ztr=embed(models,train,Otr,Ptr,dev); Zte=embed(models,test,Ote,Pte,dev)
    pca=PCA(n_components=min(32,train.shape[1],len(train)-1),random_state=seed).fit(train); Qtr=pca.transform(train); Qte=pca.transform(test)
    cmap={c:i for i,c in enumerate(classes)}; yytr=np.array([cmap[x] for x in ytr]); yyte=np.array([cmap[x] for x in yte])
    result={'n_train':int(len(ytr)),'n_test':int(len(yte)),'classes':classes,'mean_measured_genes':float(np.mean(overlap))}
    for name,A,B in [('maskaware',Ztr,Zte),('raw',train,test),('pca',Qtr,Qte)]:
        clf=LogisticRegression(max_iter=250,C=1.0,class_weight='balanced',random_state=seed,n_jobs=2)
        clf.fit(A,yytr); pred=clf.predict(B)
        pf=f1_score(yyte,pred,average=None,labels=np.arange(len(classes)))
        rngb=np.random.default_rng(seed+991)
        boots=[]
        for _ in range(bootstrap):
            bi=rngb.integers(0,len(yyte),size=len(yyte)); boots.append(float(f1_score(yyte[bi],pred[bi],average='macro',labels=np.arange(len(classes)))))
        per_file={}
        for fi,fn in enumerate(test_names):
            mk=test_file_id==fi
            if mk.any(): per_file[fn]={'macro_f1':float(f1_score(yyte[mk],pred[mk],average='macro',labels=np.arange(len(classes)) )),'n':int(mk.sum())}
        result[name]={'macro_f1':float(f1_score(yyte,pred,average='macro')),'accuracy':float(np.mean(pred==yyte)),'per_class_f1':{c:float(v) for c,v in zip(classes,pf)},'per_file':per_file,'bootstrap95':[float(np.quantile(boots,0.025)),float(np.quantile(boots,0.975))]}
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-dir',required=True); ap.add_argument('--pattern',default='*_level0.npz'); ap.add_argument('--label-name',default='annot_level0'); ap.add_argument('--checkpoint',action='append',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seed',type=int,default=20262016); ap.add_argument('--max-per-file',type=int,default=6000); ap.add_argument('--bootstrap',type=int,default=200); a=ap.parse_args()
    rows=sorted(glob.glob(str(Path(a.cache_dir)/a.pattern))); names=[os.path.basename(x) for x in rows]
    if len(rows)<8: raise RuntimeError(f'expected 8 caches, found {len(rows)}')
    b=np.load(rows[0],allow_pickle=False); genes=b['genes'].astype(str); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); models=[]
    for ck in a.checkpoint:
        o=torch.load(ck,map_location='cpu'); c=o['config']; m=MaskAware(len(genes),len(c['program_names']),c['hidden'],c['latent'],c.get('platform_dim',8)).to(dev); m.load_state_dict(o['model'],strict=False); m.eval(); models.append(m)
    # Fixed, accession-level splits; no cell-level leakage.
    split1=(names[:4],names[4:]); split2=(names[4:],names[:4]); res=[]
    for i,(tr,te) in enumerate([split1,split2]): res.append({'split':i,'train_files':tr,'test_files':te,'metrics':run_split(rows,tr,te,models,genes,a.seed+i,a.max_per_file,dev,a.bootstrap)})
    out={'stage':'novae_label','label':a.label_name,'n_files':len(rows),'checkpoints':a.checkpoint,'splits':res,'note':'Independent public Novae/Xenium annotations; 23-gene overlap is retained as a strict targeted-panel condition. Metrics are donor/accession-heldout linear probes, not zero-shot labels.'}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
