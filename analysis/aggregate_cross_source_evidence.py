#!/usr/bin/env python3
"""Freeze a cross-source evidence table for the boundary-study manuscript."""
import argparse,json
from pathlib import Path

def read(p): return json.loads(Path(p).read_text())
def rec(source,task,unit,model,raw,pca,status,note):
    return {'source':source,'task':task,'statistical_unit':unit,'model':model,'raw':raw,'pca':pca,'status':status,'note':note}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); r=Path(a.root); rows=[]
    x=read(r/'reports/label_heldout_train/novae_label_probe_heldout.json')
    for f in x.get('splits',x.get('folds',[])):
        m=f.get('metrics',f); rows.append(rec('Novae/Xenium','coarse independent label probe', 'reverse slice split', m.get('maskaware',{}).get('macro_f1'),m.get('raw',{}).get('macro_f1'),m.get('pca',{}).get('macro_f1'),'model-positive','Internal expression-heldout coarse labels; two splits.'))
    x=read(r/'reports/luad_region/luad_region_probe.json')
    for f in x.get('folds',[]): rows.append(rec('SpatialFusion LUAD','checkerboard region label probe','spatial block fold',f['maskaware']['macro_f1'],f['raw']['macro_f1'],f['pca']['macro_f1'],'model-negative','Independent region labels; model loses both folds.'))
    x=read(r/'reports/hest_lofo/hest_lofo.json'); rows.append(rec('HEST','complete-slice LOFO organ probe','slice',x['aggregate']['maskaware']['mean_macro_f1'],x['aggregate']['raw']['mean_macro_f1'],None,'unstable-negative','Metadata-level organ task; model below raw on mean.'))
    x=read(r/'reports/hest_sgd/hest_sgd.json'); rows.append(rec('HEST','complete-slice LOFO SGD sensitivity','slice',x['aggregate']['maskaware']['mean_macro_f1'],x['aggregate']['raw']['mean_macro_f1'],None,'probe-sensitive','Alternative probe changes pooled direction; no stable advantage.'))
    for task,path in [('donor-heldout cortical layer (default seed)','reports/spatiallibd_sgd/spatiallibd_sgd.json'),('donor-heldout cortical layer (seed 260)','reports/spatiallibd_sgd_seed260/spatiallibd_sgd_seed260.json')]:
        x=read(r/path); fs=x['folds']; rows.append(rec('spatialLIBD DLPFC',task,'donor',sum(z['maskaware']['macro_f1'] for z in fs)/len(fs),sum(z['raw']['macro_f1'] for z in fs)/len(fs),sum(z['pca']['macro_f1'] for z in fs)/len(fs),'model-negative','Three independent donors; model loses raw/PCA on every donor.'))
    for stage,path in [('layer_programs','reports/layer_programs/spatiallibd_layer_programs.json'),('spatial_boundary_k6','reports/spatial_boundary/spatiallibd_boundary.json'),('spatial_boundary_k4','reports/spatial_boundary_k4/spatiallibd_boundary_k4.json'),('spatial_boundary_k12','reports/spatial_boundary_k12/spatiallibd_boundary_k12.json'),('program_moran','reports/program_moran/spatiallibd_program_moran.json')]:
        x=read(r/path); rows.append({'source':'spatialLIBD DLPFC','task':x.get('task',stage),'statistical_unit':'donor/slice','summary':x.get('summary'),'status':'mechanism-diagnostic','note':x.get('note','')})
    out={'stage':'cross_source_evidence','purpose':'cross-source evidence freeze with continuous spatial diagnostic','rows':rows,'interpretation':{'primary_claim':'Internal coarse-label gains do not establish universal transfer; independent LUAD and DLPFC tasks provide negative boundaries, HEST is probe-sensitive.','mechanistic_claim':'DLPFC has readout-dependent spatial distortion: discrete boundary diagnostics show lower within-layer continuity and higher boundary recall, whereas continuous marker Moran error is closer to PCA than raw across k=4,6,12.','fm_status':'Current checkpoint is not a general foundation model; new objective requires a fresh external donor-heldout gate.'}}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=2,ensure_ascii=False)); print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
