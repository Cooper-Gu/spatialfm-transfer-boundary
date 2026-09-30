#!/usr/bin/env python3
"""Paired donor bootstrap for DLPFC fixed-SGD seed sensitivity."""
import argparse,json
from pathlib import Path
import numpy as np

def load(path):
    obj=json.loads(Path(path).read_text())
    rows={int(x['heldout_donor']): x for x in obj['folds']}
    return obj,rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed260',required=True); ap.add_argument('--seed160',required=True); ap.add_argument('--out',required=True); ap.add_argument('--n-bootstrap',type=int,default=10000); ap.add_argument('--seed',type=int,default=20262023); a=ap.parse_args()
    o160,r160=load(a.seed160); o260,r260=load(a.seed260); donors=sorted(set(r160)&set(r260)); rng=np.random.default_rng(a.seed)
    diffs=[]
    for d in donors: diffs.append({'donor':d,'seed160_model_minus_raw':r160[d]['maskaware']['macro_f1']-r160[d]['raw']['macro_f1'],'seed260_model_minus_raw':r260[d]['maskaware']['macro_f1']-r260[d]['raw']['macro_f1'],'seed160_model_minus_pca':r160[d]['maskaware']['macro_f1']-r160[d]['pca']['macro_f1'],'seed260_model_minus_pca':r260[d]['maskaware']['macro_f1']-r260[d]['pca']['macro_f1']})
    def boot(key):
        x=np.asarray([d[key] for d in diffs]); vals=np.asarray([x[rng.integers(0,len(x),len(x))].mean() for _ in range(a.n_bootstrap)]); return {'mean':float(x.mean()),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],'p_le_0':float(np.mean(vals<=0))}
    out={'stage':'spatiallibd_bootstrap','source':'spatialLIBD Human DLPFC Visium','donors':donors,'seed160':o160.get('seed_offset'),'seed260':o260.get('seed_offset'),'paired_donor_differences':diffs,'seed160_model_minus_raw':boot('seed160_model_minus_raw'),'seed260_model_minus_raw':boot('seed260_model_minus_raw'),'seed160_model_minus_pca':boot('seed160_model_minus_pca'),'seed260_model_minus_pca':boot('seed260_model_minus_pca'),'n_bootstrap':a.n_bootstrap,'note':'Donor-level bootstrap; three donors only, interpreted as directional robustness evidence.'}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
