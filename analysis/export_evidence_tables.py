#!/usr/bin/env python3
"""Export frozen cross_source_evidence evidence JSON to reviewer-readable CSV tables."""
import argparse,csv,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out-dir',required=True); a=ap.parse_args(); d=json.loads(Path(a.input).read_text()); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    with (out/'cross_source_rows.csv').open('w',newline='',encoding='utf-8-sig') as f:
        cols=['source','task','statistical_unit','model','raw','pca','status','note']; w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for r in d['rows']:
            if 'summary' in r:
                w.writerow({'source':r['source'],'task':r['task'],'statistical_unit':r['statistical_unit'],'model':json.dumps(r['summary'].get('model'),ensure_ascii=False),'raw':json.dumps(r['summary'].get('raw'),ensure_ascii=False),'pca':json.dumps(r['summary'].get('pca'),ensure_ascii=False),'status':r['status'],'note':r['note']})
            else:w.writerow({k:r.get(k) for k in cols})
    with (out/'mechanism_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        cols=['source','task','statistical_unit','status','model_summary','raw_summary','pca_summary','note']; w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for r in d['rows']:
            if 'summary' in r:w.writerow({'source':r['source'],'task':r['task'],'statistical_unit':r['statistical_unit'],'status':r['status'],'model_summary':json.dumps(r['summary'].get('model'),ensure_ascii=False),'raw_summary':json.dumps(r['summary'].get('raw'),ensure_ascii=False),'pca_summary':json.dumps(r['summary'].get('pca'),ensure_ascii=False),'note':r['note']})
    print(json.dumps({'out_dir':str(out),'files':['cross_source_rows.csv','mechanism_summary.csv'],'n_rows':len(d['rows'])},ensure_ascii=False))
if __name__=='__main__': main()
