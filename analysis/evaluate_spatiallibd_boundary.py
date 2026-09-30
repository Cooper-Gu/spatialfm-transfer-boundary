#!/usr/bin/env python3
"""Spatial boundary/within-layer consistency on DLPFC donor-heldout predictions."""
import argparse,glob,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors

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

def fit_predict(A,y,B,seed):
    mu=A.mean(0); sd=A.std(0); sd[sd<1e-6]=1.; A=(A-mu)/sd; B=(B-mu)/sd; clf=SGDClassifier(loss='log_loss',alpha=1e-4,max_iter=100,tol=1e-2,average=True,class_weight='balanced',random_state=seed,n_jobs=2); clf.fit(A,y); return clf.predict(B)

def fit_clf(A,y,seed):
    mu=A.mean(0); sd=A.std(0); sd[sd<1e-6]=1.; clf=SGDClassifier(loss='log_loss',alpha=1e-4,max_iter=100,tol=1e-2,average=True,class_weight='balanced',random_state=seed,n_jobs=2); clf.fit((A-mu)/sd,y); return clf,mu,sd

def predict_clf(model,B):
    clf,mu,sd=model; return clf.predict((B-mu)/sd)

def edge_metrics(coord,y,pred,k=6):
    nnx=NearestNeighbors(n_neighbors=min(k+1,len(coord)),algorithm='kd_tree').fit(coord); ind=nnx.kneighbors(return_distance=False)[:,1:]; ii=np.repeat(np.arange(len(coord)),ind.shape[1]); jj=ind.reshape(-1); keep=ii<jj; ii=ii[keep]; jj=jj[keep]; ts=(y[ii]==y[jj]); ps=(pred[ii]==pred[jj]);
    return {'n_edges':int(len(ii)),'true_same_edge_fraction':float(ts.mean()),'pred_same_edge_fraction':float(ps.mean()),'within_layer_consistency':float(ps[ts].mean()) if np.any(ts) else None,'boundary_recall':float((~ps[~ts]).mean()) if np.any(~ts) else None,'edge_agreement_accuracy':float((ps==ts).mean())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-dir',required=True); ap.add_argument('--checkpoint',action='append',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seed-offset',type=int,default=266); ap.add_argument('--k',type=int,default=6); a=ap.parse_args()
    fs=sorted(glob.glob(str(Path(a.cache_dir)/'*.npz'))); dm={sid:i for i,g in enumerate([['151507','151508','151509','151510'],['151669','151670','151671','151672'],['151673','151674','151675','151676']]) for sid in g}; rows=[]
    for f in fs:
        z=np.load(f,allow_pickle=False); sid=Path(f).stem; rows.append({'file':sid,'donor':dm[sid],'X':z['E'].astype('float32'),'coord':z['coord'].astype('float32'),'O':np.repeat(z['panel'].astype('float32'),len(z['E']),0),'P':np.eye(8,dtype='float32')[[0]*len(z['E'])],'y':z['y'].astype(str)})
    genes=np.load(fs[0],allow_pickle=False)['genes'].astype(str); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ms=[]
    for ck in a.checkpoint:
        o=torch.load(ck,map_location='cpu'); c=o['config']; m=M(len(genes),len(c['program_names']),c['hidden'],c['latent'],c.get('platform_dim',8)).to(dev); m.load_state_dict(o['model'],strict=False); m.eval(); ms.append(m)
    for r in rows:r['Z']=emb(ms,r['X'],r['O'],r['P'],dev)
    folds=[]
    for held in [0,1,2]:
        te=[r for r in rows if r['donor']==held]; tr=[r for r in rows if r['donor']!=held]; Xtr=np.concatenate([r['X'] for r in tr]); Ztr=np.concatenate([r['Z'] for r in tr]); ytr=np.concatenate([r['y'] for r in tr]); classes=sorted(set(ytr)&set(np.concatenate([r['y'] for r in te]))); keep=np.isin(ytr,classes); Xtr=Xtr[keep]; Ztr=Ztr[keep]; ytr=ytr[keep]; cmap={x:i for i,x in enumerate(classes)}; aa=np.array([cmap[x] for x in ytr]); pca=PCA(n_components=32,svd_solver='randomized',iterated_power=2,random_state=a.seed_offset+held).fit(Xtr); Ptr=pca.transform(Xtr); raw_idx=np.argsort(-Xtr.var(0))[:min(2000,Xtr.shape[1])]; fold={'heldout_donor':held,'slices':[]}
        models={'model':fit_clf(Ztr,aa,a.seed_offset+held),'raw':fit_clf(Xtr[:,raw_idx],aa,a.seed_offset+held),'pca':fit_clf(Ptr,aa,a.seed_offset+held)}
        for r in te:
            keepte=np.isin(r['y'],classes); Xte=r['X'][keepte]; Zte=r['Z'][keepte]; yte=r['y'][keepte]; coord=r['coord'][keepte]; bb=np.array([cmap[x] for x in yte]); preds={'model':predict_clf(models['model'],Zte),'raw':predict_clf(models['raw'],Xte[:,raw_idx]),'pca':predict_clf(models['pca'],pca.transform(Xte))}; metrics={name:edge_metrics(coord,bb,pr,a.k) for name,pr in preds.items()}; fold['slices'].append({'file':r['file'],'n_spots':int(len(bb)),'metrics':metrics})
        folds.append(fold)
    summary={}
    for name in ['model','raw','pca']:
        vals=[s['metrics'][name] for f in folds for s in f['slices']]; summary[name]={k:float(np.mean([v[k] for v in vals])) for k in ['true_same_edge_fraction','pred_same_edge_fraction','within_layer_consistency','boundary_recall','edge_agreement_accuracy']}
    out={'stage':'spatial_boundary','source':'spatialLIBD Human DLPFC Visium','task':'spatial layer-boundary preservation','holdout':'leave-one-donor-out','k_neighbors':a.k,'folds':folds,'summary':summary,'note':'Coordinates and labels from the held-out slices; classifiers are trained on other donors only. Raw uses top-variance training genes and PCA is train-only.'}; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
