#!/usr/bin/env python3
"""Fast fixed-SGD donor-heldout gate on spatialLIBD DLPFC."""
import argparse,glob,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
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
def balanced_idx(y,max_per,seed):
    rng=np.random.default_rng(seed); take=[]
    for c in np.unique(y):
        q=np.flatnonzero(y==c); take.extend(rng.choice(q,min(max_per,len(q)),replace=False))
    take=np.asarray(take); rng.shuffle(take); return take
def fit(A,y,B,seed):
    mu=A.mean(0); sd=A.std(0); sd[sd<1e-6]=1.; A=(A-mu)/sd; B=(B-mu)/sd; idx=balanced_idx(y,1500,seed); A=A[idx]; y=y[idx]
    clf=SGDClassifier(loss='log_loss',alpha=1e-4,max_iter=100,tol=1e-2,average=True,class_weight='balanced',random_state=seed,n_jobs=2); clf.fit(A,y); return clf.predict(B)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-dir',required=True); ap.add_argument('--checkpoint',action='append',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seed-offset',type=int,default=160); ap.add_argument('--stage',default='spatiallibd_sgd'); ap.add_argument('--permute-train-labels',action='store_true'); a=ap.parse_args(); fs=sorted(glob.glob(str(Path(a.cache_dir)/'*.npz'))); dm={sid:i for i,g in enumerate([['151507','151508','151509','151510'],['151669','151670','151671','151672'],['151673','151674','151675','151676']]) for sid in g}; rows=[]
    for f in fs:
        z=np.load(f,allow_pickle=False); sid=Path(f).stem; rows.append({'file':sid,'donor':dm[sid],'X':z['E'].astype('float32'),'O':np.repeat(z['panel'].astype('float32'),len(z['E']),0),'P':np.eye(8,dtype='float32')[[0]*len(z['E'])],'y':z['y'].astype(str)})
    genes=np.load(fs[0],allow_pickle=False)['genes'].astype(str); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ms=[]
    for ck in a.checkpoint:
        o=torch.load(ck,map_location='cpu'); c=o['config']; m=M(len(genes),len(c['program_names']),c['hidden'],c['latent'],c.get('platform_dim',8)).to(dev); m.load_state_dict(o['model'],strict=False); m.eval(); ms.append(m)
    for r in rows:r['Z']=emb(ms,r['X'],r['O'],r['P'],dev)
    folds=[]
    for held in [0,1,2]:
        te=[r for r in rows if r['donor']==held]; tr=[r for r in rows if r['donor']!=held]; Xtr=np.concatenate([r['X'] for r in tr]); Ztr=np.concatenate([r['Z'] for r in tr]); ytr=np.concatenate([r['y'] for r in tr]); Xte=np.concatenate([r['X'] for r in te]); Zte=np.concatenate([r['Z'] for r in te]); yte=np.concatenate([r['y'] for r in te]); classes=np.array(sorted(set(ytr)&set(yte))); keep=np.isin(ytr,classes); keepte=np.isin(yte,classes); Xtr=Xtr[keep]; Ztr=Ztr[keep]; ytr=ytr[keep]; Xte=Xte[keepte]; Zte=Zte[keepte]; classes=classes.tolist(); cmap={x:i for i,x in enumerate(classes)}; aa=np.array([cmap[x] for x in ytr]); bb=np.array([cmap[x] for x in yte[keepte]]); seed=a.seed_offset+held; idx=balanced_idx(ytr,1500,seed); pca=PCA(n_components=min(32,len(idx)-1,Xtr.shape[1]),svd_solver='randomized',iterated_power=2,random_state=seed).fit(Xtr[idx]); row={'heldout_donor':held,'test_files':[r['file'] for r in te],'n_train':int(len(ytr)),'n_test':int(len(bb)),'classes':classes,'probe_seed':seed,'permuted_train_labels':bool(a.permute_train_labels)}; aa=np.random.default_rng(seed+100000).permutation(aa) if a.permute_train_labels else aa
        for name,A,B in [('maskaware',Ztr,Zte),('raw',Xtr,Xte),('pca',pca.transform(Xtr),pca.transform(Xte))]:
            pr=fit(A,aa,B,seed)
            row[name]={'macro_f1':float(f1_score(bb,pr,average='macro',labels=np.arange(len(classes)),zero_division=0)),'accuracy':float(np.mean(pr==bb))}
        folds.append(row)
    out={'stage':a.stage,'source':'spatialLIBD Human DLPFC Visium','task':'spot-level cortical layer','holdout':'leave-one-donor-out (3 donors, 12 slices)','common_genes':5442,'seed_offset':a.seed_offset,'folds':folds,'note':'Frozen label_heldout checkpoints; fixed 100-step SGD, 1500 train spots/class cap, no target-donor adaptation; train-only randomized PCA.'}; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
