#!/usr/bin/env python3
"""Build independent HEST ST wide-panel caches with metadata organ labels."""
import argparse,json
from pathlib import Path
import anndata as ad
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--h5ad',required=True); ap.add_argument('--backbone',required=True); ap.add_argument('--label',required=True); ap.add_argument('--out',required=True); ap.add_argument('--max-rows',type=int,default=2000); a=ap.parse_args()
    b=np.load(a.backbone,allow_pickle=False); genes=b['genes'].astype(str); gm={g.upper():i for i,g in enumerate(genes)}; x=ad.read_h5ad(a.h5ad); vals=x.var_names.astype(str).tolist(); pairs=[]; seen=set()
    for si,g in enumerate(vals):
      di=gm.get(g.upper());
      if di is not None and di not in seen: pairs.append((si,di)); seen.add(di)
    if len(pairs)<1000: raise RuntimeError(f'overlap {len(pairs)} < 1000')
    rng=np.random.default_rng(149); idx=np.arange(x.n_obs); idx=rng.choice(idx,size=min(a.max_rows,len(idx)),replace=False); src=np.asarray([p[0] for p in pairs]); dst=np.asarray([p[1] for p in pairs]); raw=x[idx].X; raw=raw.toarray() if hasattr(raw,'toarray') else np.asarray(raw); raw=np.nan_to_num(raw[:,src].astype('float32')); lib=raw.sum(1,keepdims=True); raw=np.log1p(raw/np.maximum(lib,1)*1e4); E=np.zeros((len(idx),len(genes)),dtype='float16'); E[:,dst]=raw.astype('float16'); panel=np.zeros(len(genes),dtype='uint8'); panel[dst]=1; coord=np.zeros((len(idx),2),dtype='float32')
    for k in x.obsm_keys():
      if 'spatial' in str(k).lower():
       q=np.asarray(x.obsm[k]);
       if q.ndim==2 and q.shape[1]>=2: coord=q[idx,:2].astype('float32'); break
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,E=E,coord=coord,asset=np.zeros(len(E),dtype='int16'),donor=np.zeros(len(E),dtype='int32'),panel=panel[None,:],genes=genes,asset_abbr=np.asarray([out.stem]),asset_donor=np.asarray([0],dtype='int32'),asset_platform=np.asarray([0],dtype='int16'),region_label=np.asarray([a.label]*len(E)))
    meta={'h5ad':a.h5ad,'label':a.label,'n_rows':int(len(E)),'n_genes':int(len(genes)),'overlap_genes':int(len(pairs))}; out.with_suffix('.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
