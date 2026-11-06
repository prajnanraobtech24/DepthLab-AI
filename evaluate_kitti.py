"""Evaluate classical sparse-depth completion on prepared KITTI split (metres)."""
import argparse, glob, os, json, time, statistics
import numpy as np
from src.pipeline import reconstruct
from src.metrics import compute_metrics


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/processed/kitti'); p.add_argument('--split',default='val',choices=['train','val','test']); p.add_argument('--out',default='results/kitti_evaluation.json'); p.add_argument('--limit',type=int,default=0); p.add_argument('--repeats',type=int,default=3); a=p.parse_args()
    root=f'{a.data}/{a.split}'; rg={os.path.basename(x):x for x in glob.glob(root+'/rgb/*.npy')}; sp={os.path.basename(x):x for x in glob.glob(root+'/sparse/*.npy')}; gt={os.path.basename(x):x for x in glob.glob(root+'/gt/*.npy')}
    ids=sorted(set(rg)&set(sp)&set(gt)); ids=ids[:a.limit] if a.limit else ids
    if not ids: raise SystemExit(f'No complete KITTI RGB/sparse/GT triplets under {root}')
    rows=[]; methods=[('Sparse input','Interpolation','guided'),('KNN','KNN','guided'),('Guided filter','Classical','guided'),('Joint bilateral','Classical','joint_bilateral')]
    for sid in ids:
        rgb=np.load(rg[sid]); sparse=np.load(sp[sid]).astype(np.float32); truth=np.load(gt[sid]).astype(np.float32)
        valid=np.isfinite(truth)&(truth>0)&(truth<=80)
        for label,mode,filt in methods:
            result=None; times=[]
            for rep in range(a.repeats+1):
                t=time.perf_counter(); result=reconstruct(rgb,sparse,4,filt,None,mode,min_depth=.1,max_depth=80.,include_comparisons=False,depth_input_type='full_resolution_sparse'); elapsed=(time.perf_counter()-t)*1000
                if rep>0: times.append(elapsed)
            pred=.1+result['output']*79.9; pred[~result['output_valid']]=0
            rows.append({'sample':sid,'method':label,**compute_metrics(pred,truth,valid,.1,80.),'median_latency_ms':float(statistics.median(times))})
    keys=['RMSE_m','MAE_m','PSNR_dB','SSIM','Edge_F1','Boundary_F1_2px','Boundary_MAE_m','prediction_coverage','median_latency_ms']
    summary=[]
    for label,_,_ in methods:
        subset=[r for r in rows if r['method']==label]
        summary.append({'method':label,'n_samples':len(subset),**{k:float(np.mean([r[k] for r in subset if r.get(k) is not None])) if any(r.get(k) is not None for r in subset) else None for k in keys}})
    os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True)
    with open(a.out,'w') as f: json.dump({'dataset':'KITTI Depth Completion','split':a.split,'depth_units':'metres','depth_range_m':[.1,80.0],'n_samples':len(ids),'note':'Classical sparse-depth evaluation only. NYU-trained CNN is not evaluated with a mismatched 0.1-10 m normalization. Do not use hidden test ground truth.','summary':summary,'per_sample':rows},f,indent=2)
    print(json.dumps(summary,indent=2)); print(f'Wrote {a.out}')
if __name__=='__main__': main()
