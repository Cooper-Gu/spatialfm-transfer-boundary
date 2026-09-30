#!/usr/bin/env python3
"""Convert spatialLIBD 10x HDF5 + official spot labels to frozen model caches."""
import argparse,json,glob
from pathlib import Path
import h5py,numpy as np,pandas as pd
from scipy.sparse import csc_matrix

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-dir',required=True); ap.add_argument('--backbone',required=True); ap.add_argument('--out-dir',required=True); a=ap.parse_args()
    genes=np.load(a.backbone,allow_pickle=False)['genes'].astype(str); gm={g.upper():i for i,g in enumerate(genes)}; outdir=Path(a.out_dir); outdir.mkdir(parents=True,exist_ok=True); metas=[]
    for hp in sorted(glob.glob(str(Path(a.raw_dir)/'*_filtered_feature_bc_matrix.h5'))):
        sid=Path(hp).name.split('_')[0]; mp=Path(a.raw_dir)/(sid+'_metadata.tsv'); m=pd.read_csv(mp,sep='\t');
        with h5py.File(hp,'r') as f:
            g=f['matrix']; shape=tuple(map(int,g['shape'][:])); names=[x.decode() for x in g['features/name'][:]]; bars=[x.decode() for x in g['barcodes'][:]]; data=g['data'][:]; idx=g['indices'][:]; indptr=g['indptr'][:]
        pairs=[]; seen=set()
        for si,name in enumerate(names):
            di=gm.get(name.upper())
            if di is not None and di not in seen: pairs.append((si,di)); seen.add(di)
        if len(pairs)<1000: raise RuntimeError(f'{sid}: overlap {len(pairs)} < 1000')
        bm={b:i for i,b in enumerate(bars)}; m['barcode']=m['barcode'].astype(str); mi=np.array([bm.get(b,-1) for b in m['barcode']],dtype=int); valid=(mi>=0) & (~m['discard'].astype(bool)) & (~m['layer_guess_reordered_short'].isna())
        m=m.loc[valid].copy(); mi=mi[valid]; labels=m['layer_guess_reordered_short'].astype(str).to_numpy(); order=np.argsort(mi); m=m.iloc[order].reset_index(drop=True); mi=mi[order]; labels=np.asarray(labels[order],dtype='<U8')
        Xfull=csc_matrix((data,idx,indptr),shape=shape).tocsr(); X=Xfull[np.array([p[0] for p in pairs]),:][:,mi].T.tocsr(); raw=X.toarray().astype('float32'); lib=raw.sum(1,keepdims=True); raw=np.log1p(raw/np.maximum(lib,1)*1e4)
        E=np.zeros((len(raw),len(genes)),dtype='float16'); dst=np.array([p[1] for p in pairs]); E[:,dst]=raw.astype('float16'); panel=np.zeros(len(genes),dtype='uint8'); panel[dst]=1; coord=m[['imagerow','imagecol']].to_numpy(dtype='float32')
        out=outdir/f'{sid}.npz'; np.savez_compressed(out,E=E,coord=coord,asset=np.zeros(len(E),dtype='int16'),donor=np.zeros(len(E),dtype='int32'),panel=panel[None,:],genes=genes,asset_abbr=np.asarray([sid]),asset_donor=np.asarray([0],dtype='int32'),asset_platform=np.asarray([0],dtype='int16'),y=labels)
        meta={'sample':sid,'n_spots':int(len(E)),'overlap_genes':int(len(pairs)),'label_counts':{str(k):int(v) for k,v in pd.Series(labels).value_counts().items()}}; (out.with_suffix('.json')).write_text(json.dumps(meta,indent=2,ensure_ascii=False)); metas.append(meta); print(json.dumps(meta,ensure_ascii=False),flush=True)
    (outdir/'manifest.json').write_text(json.dumps({'stage':'spatiallibd_cache','source':'spatialLIBD Human DLPFC Visium','samples':metas},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
