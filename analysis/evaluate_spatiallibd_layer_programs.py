#!/usr/bin/env python3
"""Train-donor-derived cortical layer marker program transfer on DLPFC."""
import argparse,glob,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

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

def standardize(A,B):
    mu=A.mean(0); sd=A.std(0); sd[sd<1e-6]=1.; return (A-mu)/sd,(B-mu)/sd

def marker_programs(X,y,classes,n_markers=25):
    mu=X.mean(0); var=X.var(0)+1e-4; out={}
    for c in classes:
        a=X[y==c]; b=X[y!=c]; d=(a.mean(0)-b.mean(0))/np.sqrt(var); idx=np.argsort(-np.abs(d))[:n_markers]; out[c]=idx.tolist()
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-dir',required=True); ap.add_argument('--checkpoint',action='append',required=True); ap.add_argument('--out',required=True); ap.add_argument('--n-markers',type=int,default=25); a=ap.parse_args()
    fs=sorted(glob.glob(str(Path(a.cache_dir)/'*.npz'))); dm={sid:i for i,g in enumerate([['151507','151508','151509','151510'],['151669','151670','151671','151672'],['151673','151674','151675','151676']]) for sid in g}; rows=[]
    for f in fs:
        z=np.load(f,allow_pickle=False); sid=Path(f).stem; rows.append({'file':sid,'donor':dm[sid],'X':z['E'].astype('float32'),'O':np.repeat(z['panel'].astype('float32'),len(z['E']),0),'P':np.eye(8,dtype='float32')[[0]*len(z['E'])],'y':z['y'].astype(str)})
    genes=np.load(fs[0],allow_pickle=False)['genes'].astype(str); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ms=[]
    for ck in a.checkpoint:
        o=torch.load(ck,map_location='cpu'); c=o['config']; m=M(len(genes),len(c['program_names']),c['hidden'],c['latent'],c.get('platform_dim',8)).to(dev); m.load_state_dict(o['model'],strict=False); m.eval(); ms.append(m)
    for r in rows:r['Z']=emb(ms,r['X'],r['O'],r['P'],dev)
    folds=[]
    for held in [0,1,2]:
        te=[r for r in rows if r['donor']==held]; tr=[r for r in rows if r['donor']!=held]; Xtr=np.concatenate([r['X'] for r in tr]); Ztr=np.concatenate([r['Z'] for r in tr]); ytr=np.concatenate([r['y'] for r in tr]); Xte=np.concatenate([r['X'] for r in te]); Zte=np.concatenate([r['Z'] for r in te]); yte=np.concatenate([r['y'] for r in te]); classes=sorted(set(ytr)&set(yte)); keep=np.isin(ytr,classes); keepte=np.isin(yte,classes); Xtr=Xtr[keep]; Ztr=Ztr[keep]; ytr=ytr[keep]; Xte=Xte[keepte]; Zte=Zte[keepte]; yte=yte[keepte]; pca=PCA(n_components=32,svd_solver='randomized',iterated_power=2,random_state=165+held).fit(Xtr); Ptr=pca.transform(Xtr); Pte=pca.transform(Xte); raw_idx=np.argsort(-Xtr.var(0))[:min(2000,Xtr.shape[1])]; Atr={'model':Ztr,'raw':Xtr[:,raw_idx],'pca':Ptr}; Bte={'model':Zte,'raw':Xte[:,raw_idx],'pca':Pte}; markers=marker_programs(Xtr,ytr,classes,a.n_markers); scores=[]
        for c in classes:
            idx=np.asarray(markers[c]); trscore=Xtr[:,idx].mean(1); tescore=Xte[:,idx].mean(1); row={'class':c,'n_markers':int(len(idx)),'markers':genes[idx].tolist()}
            for name in Atr:
                aa,bb=standardize(Atr[name],Bte[name]); clf=Ridge(alpha=10.0,solver='lsqr').fit(aa,trscore); pred=clf.predict(bb); row[name]={'pearson_r':float(pearsonr(tescore,pred).statistic),'r2':float(1-np.sum((tescore-pred)**2)/np.sum((tescore-tescore.mean())**2))}
            scores.append(row)
        folds.append({'heldout_donor':held,'test_files':[r['file'] for r in te],'classes':classes,'programs':scores[-len(classes):]})
    summary={}
    for name in ['model','raw','pca']:
        vals=[q[name]['pearson_r'] for f in folds for q in f['programs']]; summary[name]={'mean_pearson_r':float(np.mean(vals)),'median_pearson_r':float(np.median(vals)),'n':len(vals)}
    out={'stage':'layer_programs','source':'spatialLIBD Human DLPFC Visium','task':'train-donor-derived cortical layer marker program transfer','holdout':'leave-one-donor-out','n_markers_per_layer':a.n_markers,'folds':folds,'summary':summary,'note':'Marker genes are selected only from the two training donors in each fold; no target-donor labels are used for marker selection or probe fitting.'}; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
