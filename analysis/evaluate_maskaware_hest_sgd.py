#!/usr/bin/env python3
"""Fast fixed-iteration SGD sensitivity check for HEST slice LOFO."""
import argparse,glob,json,os
from pathlib import Path
import numpy as np, torch
from torch import nn
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
def subsample(X,y,max_per=1500,seed=154):
    rng=np.random.default_rng(seed); take=[]
    for c in np.unique(y):
        q=np.flatnonzero(y==c); take.extend(rng.choice(q,min(max_per,len(q)),replace=False))
    take=np.array(take); rng.shuffle(take); return X[take],y[take]
def fit_predict(Atr,ytr,B,seed):
    mu=Atr.mean(0); sd=Atr.std(0); sd[sd<1e-6]=1.; Atr=(Atr-mu)/sd; B=(B-mu)/sd; Atr,ytr=subsample(Atr,ytr)
    clf=SGDClassifier(loss='log_loss',alpha=1e-4,max_iter=100,tol=1e-2,average=True,class_weight='balanced',random_state=seed,n_jobs=2); clf.fit(Atr,ytr); return clf.predict(B)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-dir',required=True); ap.add_argument('--checkpoint',action='append',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); fs=sorted(glob.glob(str(Path(a.cache_dir)/'*.npz'))); rows=[]
    for f in fs:
        z=np.load(f,allow_pickle=False); rows.append({'file':os.path.basename(f),'X':z['E'].astype('float32'),'O':np.repeat(z['panel'].astype('float32'),len(z['E']),0),'P':np.eye(8,dtype='float32')[[0]*len(z['E'])],'y':str(z['region_label'].astype(str)[0])})
    genes=np.load(fs[0],allow_pickle=False)['genes'].astype(str); d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ms=[]
    for ck in a.checkpoint:
        o=torch.load(ck,map_location='cpu'); c=o['config']; m=M(len(genes),len(c['program_names']),c['hidden'],c['latent'],c.get('platform_dim',8)).to(d); m.load_state_dict(o['model'],strict=False); m.eval(); ms.append(m)
    for r in rows:r['Z']=emb(ms,r['X'],r['O'],r['P'],d)
    labels=sorted({r['y'] for r in rows}); cmap={c:i for i,c in enumerate(labels)}; folds=[]
    for ti,test in enumerate(rows):
        tr=[r for j,r in enumerate(rows) if j!=ti]; Xtr=np.concatenate([r['X'] for r in tr]); Ztr=np.concatenate([r['Z'] for r in tr]); ytr=np.array([cmap[r['y']] for r in tr for _ in range(len(r['X']))]); yte=np.full(len(test['X']),cmap[test['y']]); row={'test_file':test['file'],'label':test['y'],'n_test':int(len(yte))}
        for n,A,B in [('maskaware',Ztr,test['Z']),('raw',Xtr,test['X'])]:
            pr=fit_predict(A,ytr,B,154+ti); row[n]={'macro_f1':float(f1_score(yte,pr,average='macro')),'accuracy':float(np.mean(pr==yte))}
        folds.append(row)
    agg={}
    for n in ['maskaware','raw']:
        v=np.array([r[n]['macro_f1'] for r in folds]); raw=np.array([r['raw']['macro_f1'] for r in folds]); agg[n]={'mean_macro_f1':float(v.mean()),'sd_macro_f1':float(v.std(ddof=1)),'wins_vs_raw':int(np.sum(v>raw))}
    out={'stage':'hest_sgd','source':'HEST ST independent wide-panel slices','n_files':len(rows),'labels':labels,'aggregate':agg,'folds':folds,'note':'Fixed 100-iteration SGD probe; at most 1,500 training spots per class per fold; complete-slice leave-one-out; metadata-level organ labels.'}; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
