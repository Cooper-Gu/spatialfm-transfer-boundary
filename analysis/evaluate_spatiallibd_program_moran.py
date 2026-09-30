#!/usr/bin/env python3
"""Label-independent spatial autocorrelation of transferred layer-marker programs."""
import argparse,glob,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
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

def marker_programs(X,y,classes,n_markers):
    var=X.var(0)+1e-4; out={}
    for c in classes:
        a=X[y==c]; b=X[y!=c]; d=(a.mean(0)-b.mean(0))/np.sqrt(var); out[c]=np.argsort(-np.abs(d))[:n_markers]
    return out

def moran(x,coord,k):
    x=np.asarray(x,float); n=len(x); xc=x-x.mean(); den=float(np.sum(xc*xc));
    if den<=1e-12 or n<3:return float('nan')
    ind=NearestNeighbors(n_neighbors=min(k+1,n),algorithm='kd_tree').fit(coord).kneighbors(return_distance=False)[:,1:]
    ii=np.repeat(np.arange(n),ind.shape[1]); jj=ind.reshape(-1); w=len(ii); num=float(np.sum(xc[ii]*xc[jj])); return float(n/w*num/den)

def zfit(A,B):
    mu=A.mean(0); sd=A.std(0); sd[sd<1e-6]=1.; return (A-mu)/sd,(B-mu)/sd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-dir',required=True); ap.add_argument('--checkpoint',action='append',required=True); ap.add_argument('--out',required=True); ap.add_argument('--n-markers',type=int,default=25); ap.add_argument('--ks',default='4,6,12'); a=ap.parse_args(); ks=[int(x) for x in a.ks.split(',')]
    fs=sorted(glob.glob(str(Path(a.cache_dir)/'*.npz'))); dm={sid:i for i,g in enumerate([['151507','151508','151509','151510'],['151669','151670','151671','151672'],['151673','151674','151675','151676']]) for sid in g}; rows=[]
    for f in fs:
        z=np.load(f,allow_pickle=False); sid=Path(f).stem; rows.append({'file':sid,'donor':dm[sid],'X':z['E'].astype('float32'),'coord':z['coord'].astype('float32'),'O':np.repeat(z['panel'].astype('float32'),len(z['E']),0),'P':np.eye(8,dtype='float32')[[0]*len(z['E'])],'y':z['y'].astype(str)})
    genes=np.load(fs[0],allow_pickle=False)['genes'].astype(str); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ms=[]
    for ck in a.checkpoint:
        o=torch.load(ck,map_location='cpu'); c=o['config']; m=M(len(genes),len(c['program_names']),c['hidden'],c['latent'],c.get('platform_dim',8)).to(dev); m.load_state_dict(o['model'],strict=False); m.eval(); ms.append(m)
    for r in rows:r['Z']=emb(ms,r['X'],r['O'],r['P'],dev)
    folds=[]
    for held in [0,1,2]:
        te=[r for r in rows if r['donor']==held]; tr=[r for r in rows if r['donor']!=held]; Xtr=np.concatenate([r['X'] for r in tr]); Ztr=np.concatenate([r['Z'] for r in tr]); ytr=np.concatenate([r['y'] for r in tr]); classes=sorted(set(ytr)&set(np.concatenate([r['y'] for r in te]))); keep=np.isin(ytr,classes); Xtr=Xtr[keep]; Ztr=Ztr[keep]; ytr=ytr[keep]; markers=marker_programs(Xtr,ytr,classes,a.n_markers); Ytr=np.column_stack([Xtr[:,markers[c]].mean(1) for c in classes]); pca=PCA(n_components=32,svd_solver='randomized',iterated_power=2,random_state=171+held).fit(Xtr); Ptr=pca.transform(Xtr); raw_idx=np.argsort(-Xtr.var(0))[:min(2000,Xtr.shape[1])]; Atr={'model':Ztr,'raw':Xtr[:,raw_idx],'pca':Ptr}; fold={'heldout_donor':held,'programs':[]}
        fitted={}
        for name,A in Atr.items():
            mu=A.mean(0); sd=A.std(0); sd[sd<1e-6]=1.; clf=Ridge(alpha=10.0,solver='lsqr').fit((A-mu)/sd,Ytr); fitted[name]=(clf,mu,sd)
        for r in te:
            Yte=np.column_stack([r['X'][:,markers[c]].mean(1) for c in classes]); coords=r['coord']; item={'file':r['file'],'n_spots':int(len(coords)),'programs':[]}
            for j,c in enumerate(classes):
                q={'class':c,'markers':genes[markers[c]].tolist()}; true_i={k:moran(Yte[:,j],coords,k) for k in ks}; q['true_moran_i']=true_i
                for name,A in Atr.items():
                    if name=='model':B=r['Z']
                    elif name=='raw':B=r['X'][:,raw_idx]
                    else:B=pca.transform(r['X'])
                    clf,mu,sd=fitted[name]; pred=clf.predict((B-mu)/sd)[:,j]; pred_i={k:moran(pred,coords,k) for k in ks}; q[name]={'pearson_r':float(pearsonr(Yte[:,j],pred).statistic),'moran_i':pred_i,'abs_moran_error':{str(k):float(abs(pred_i[k]-true_i[k])) for k in ks}}
                item['programs'].append(q)
            fold['programs'].append(item)
        folds.append(fold)
    summary={}
    for name in ['model','raw','pca']:
        summary[name]={}
        for k in ks:
            vals=[]; signed=[]; corr=[]
            for f in folds:
                for sl in f['programs']:
                    for q in sl['programs']:
                        vals.append(q[name]['abs_moran_error'][str(k)]); signed.append(q[name]['moran_i'][k]-q['true_moran_i'][k]); corr.append(q[name]['pearson_r'])
            summary[name][str(k)]={'mean_abs_moran_error':float(np.mean(vals)),'mean_signed_moran_error':float(np.mean(signed)),'mean_pearson_r':float(np.mean(corr)),'n':len(vals)}
    out={'stage':'program_moran','source':'spatialLIBD Human DLPFC Visium','task':'label-independent spatial autocorrelation of transferred layer-marker programs','holdout':'leave-one-donor-out','n_markers_per_layer':a.n_markers,'ks':ks,'folds':folds,'summary':summary,'note':'Marker sets and ridge maps are fit only on training donors; target-donor layer labels are not used. Moran I is computed on target coordinates and continuous predicted module scores.'}; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
