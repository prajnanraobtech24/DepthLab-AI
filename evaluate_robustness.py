"""Robustness experiments on held-out NYU samples with fixed conditions."""
import argparse, glob, os, json, cv2, numpy as np
from src.pipeline import reconstruct
from src.metrics import compute_metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/processed/nyu'); p.add_argument('--checkpoint',default='checkpoints/depthrefinenet.pt'); p.add_argument('--out',default='results/robustness.json'); p.add_argument('--limit',type=int,default=50); p.add_argument('--seed',type=int,default=123); a=p.parse_args()
    if not os.path.isfile(a.checkpoint): raise SystemExit('A trained checkpoint is required. Train first.')
    root=f'{a.data}/test'; rg={os.path.basename(x):x for x in glob.glob(root+'/rgb/*.npy')}; gt={os.path.basename(x):x for x in glob.glob(root+'/gt/*.npy')}; ids=sorted(set(rg)&set(gt))[:a.limit]
    if not ids: raise SystemExit('No matching test pairs found')
    rng=np.random.default_rng(a.seed); rows=[]
    conditions=[('clean',0,0),('missing_10pct',.10,0),('missing_30pct',.30,0),('missing_50pct',.50,0),('noise_0.02m',0,.02),('noise_0.05m',0,.05)]
    methods=[('Bilinear','Interpolation'),('KNN','KNN'),('Guided','Classical'),('CNN U-Net','Deep learning'),('Hybrid','Hybrid')]
    for si,name in enumerate(ids):
        rgb=np.load(rg[name]); truth=np.load(gt[name]).astype(np.float32); h,w=truth.shape
        valid=np.isfinite(truth)&(truth>=.1)&(truth<=10)&(truth>0)
        for scale in (2,4,8):
            lh,lw=h//scale,w//scale
            den=cv2.resize(valid.astype(np.float32),(lw,lh),interpolation=cv2.INTER_AREA)
            num=cv2.resize(np.where(valid,truth,0).astype(np.float32),(lw,lh),interpolation=cv2.INTER_AREA)
            base=num/np.maximum(den,1e-6); base[den<.75]=0
            for cond,miss,noise in conditions:
                low=base.copy(); vm=low>0
                if miss: low[rng.random(low.shape)<miss]=0
                if noise:
                    vm=low>0; low[vm]+=rng.normal(0,noise,int(vm.sum())).astype(np.float32); low[vm]=np.clip(low[vm],.1,10)
                for label,mode in methods:
                    r=reconstruct(rgb,low,scale,'guided',a.checkpoint,mode,interpolation='bilinear',min_depth=.1,max_depth=10,include_comparisons=False)
                    pred=.1+r['output']*9.9; pred[~r['output_valid']]=0.0
                    rows.append({'sample':name,'scale':scale,'condition':cond,'method':label,**compute_metrics(pred,truth,valid,.1,10),'latency_ms':r['latency_ms']})
        if (si+1)%10==0: print(f'{si+1}/{len(ids)} samples')
    summary=[]
    for sc in (2,4,8):
        for cond,_,_ in conditions:
            for label,_ in methods:
                subset=[x for x in rows if x['scale']==sc and x['condition']==cond and x['method']==label]
                keys=['RMSE_m','MAE_m','PSNR_dB','SSIM','Edge_F1','Boundary_F1_2px','Boundary_MAE_m','prediction_coverage','latency_ms']
                summary.append({'scale':sc,'condition':cond,'method':label,'n':len(subset),**{k:float(np.mean([x[k] for x in subset if x.get(k) is not None])) if any(x.get(k) is not None for x in subset) else None for k in keys}})
    os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True)
    with open(a.out,'w') as f: json.dump({'dataset':'NYU Depth V2 held-out test split','seed':a.seed,'n_test_samples':len(ids),'conditions':conditions,'per_sample':rows,'summary':summary},f,indent=2)
    print(f'Wrote {a.out}')
if __name__=='__main__': main()
