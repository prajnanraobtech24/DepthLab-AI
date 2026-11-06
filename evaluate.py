"""Held-out comparison and ablations for P4. Metrics are reported in metres."""
import argparse, glob, os, json, time, statistics, hashlib, sys
import cv2, numpy as np, torch, psutil
from src.pipeline import reconstruct
from src.metrics import compute_metrics


def mean_rows(rows, keys):
    out={}
    for key in keys:
        vals=[r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        out[key]=float(np.mean(vals)) if vals else None
    out['n_samples']=len(rows)
    return out


def timed_run(rgb, low, scale, filt, checkpoint, mode, interp, repeats, **kwargs):
    fn=lambda: reconstruct(rgb,low,scale,filt,checkpoint,mode,interpolation=interp,min_depth=.1,max_depth=10.,include_comparisons=False,**kwargs)
    times=[]; result=None
    for _ in range(max(1, repeats)):
        t=time.perf_counter(); result=fn(); times.append((time.perf_counter()-t)*1000)
    return result,float(statistics.median(times))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/processed/nyu'); ap.add_argument('--checkpoint',default='checkpoints/depthrefinenet.pt'); ap.add_argument('--out',default='results/evaluation.json'); ap.add_argument('--limit',type=int,default=0); ap.add_argument('--repeats',type=int,default=3); a=ap.parse_args()
    if not os.path.isfile(a.checkpoint): raise SystemExit('Trained checkpoint missing. Train first; DL/hybrid must not use random weights.')
    root=f'{a.data}/test'; rg={os.path.basename(x):x for x in glob.glob(root+'/rgb/*.npy')}; gt={os.path.basename(x):x for x in glob.glob(root+'/gt/*.npy')}; ids=sorted(set(rg)&set(gt))
    if a.limit: ids=ids[:a.limit]
    if not ids: raise SystemExit(f'No matching test RGB/GT pairs under {root}')
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    rows=[]; ablations=[]
    specs=[('Nearest','Interpolation','nearest','guided'),('Bilinear','Interpolation','bilinear','guided'),('Bicubic','Interpolation','bicubic','guided'),('KNN','KNN','bilinear','guided'),('Guided filter','Classical','bilinear','guided'),('Joint bilateral','Classical','bilinear','joint_bilateral'),('CNN U-Net','Deep learning','bilinear','guided'),('Hybrid','Hybrid','bilinear','guided')]
    for idx,name in enumerate(ids,1):
        rgb=np.load(rg[name]); truth=np.load(gt[name]).astype(np.float32); h,w=truth.shape
        valid=np.isfinite(truth)&(truth>=.1)&(truth<=10)&(truth>0)
        for scale in (2,4,8):
            lh,lw=h//scale,w//scale
            den=cv2.resize(valid.astype(np.float32),(lw,lh),interpolation=cv2.INTER_AREA)
            num=cv2.resize(np.where(valid,truth,0).astype(np.float32),(lw,lh),interpolation=cv2.INTER_AREA)
            low=num/np.maximum(den,1e-6); low[den<.75]=0
            for label,mode,interp,filt in specs:
                r,lat=timed_run(rgb,low,scale,filt,a.checkpoint,mode,interp,a.repeats)
                pred=.1+r['output']*9.9; pred[~r['output_valid']]=0.0
                rows.append({'sample':name,'scale':scale,'method':label,'mode':mode,'interpolation':interp,'filter':filt,**compute_metrics(pred,truth,valid,.1,10),'latency_ms':lat})
        # Four-fold input ablations at 4x: full CNN, RGB off, validity-mask off.
        scale=4; lh,lw=h//scale,w//scale
        den=cv2.resize(valid.astype(np.float32),(lw,lh),interpolation=cv2.INTER_AREA)
        num=cv2.resize(np.where(valid,truth,0).astype(np.float32),(lw,lh),interpolation=cv2.INTER_AREA)
        low=num/np.maximum(den,1e-6); low[den<.75]=0
        for label,kwargs in [('CNN full',{}),('CNN RGB off',{'use_rgb_guidance':False}),('CNN validity mask off',{'use_valid_mask':False})]:
            r,lat=timed_run(rgb,low,4,'guided',a.checkpoint,'Deep learning','bilinear',a.repeats,**kwargs)
            pred=.1+r['output']*9.9; pred[~r['output_valid']]=0.0
            ablations.append({'sample':name,'variant':label,**compute_metrics(pred,truth,valid,.1,10),'latency_ms':lat})
        print(f'Evaluated {idx}/{len(ids)}: {name}', flush=True)
    metric_keys=['RMSE_m','MAE_m','PSNR_dB','SSIM','Edge_F1','Boundary_F1_2px','Boundary_MAE_m','NonBoundary_MAE_m','prediction_coverage','latency_ms']
    summary=[]
    for scale in (2,4,8):
        for label,_,_,_ in specs:
            subset=[r for r in rows if r['scale']==scale and r['method']==label]
            summary.append({'scale':scale,'method':label,**mean_rows(subset,metric_keys)})
    ablation_summary=[]
    for label in ('CNN full','CNN RGB off','CNN validity mask off'):
        subset=[r for r in ablations if r['variant']==label]
        ablation_summary.append({'variant':label,**mean_rows(subset,metric_keys)})
    process=psutil.Process()
    try:
        import resource
        raw_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_cpu_rss_mb=raw_peak/(1024*1024 if sys.platform=='darwin' else 1024)
    except (ImportError, AttributeError): peak_cpu_rss_mb=None
    from src.model import DepthRefineNet
    parameter_count=sum(int(x.numel()) for x in DepthRefineNet().parameters())
    with open(a.checkpoint,'rb') as cf: checkpoint_sha256=hashlib.sha256(cf.read()).hexdigest()
    report={'dataset':'NYU Depth V2 held-out test split','depth_units':'metres','depth_range_m':[.1,10.0],
            'n_test_samples':len(ids),'device':'cuda' if torch.cuda.is_available() else 'cpu',
            'model_checkpoint':os.path.abspath(a.checkpoint),'checkpoint_sha256':checkpoint_sha256,
            'model_parameters':parameter_count,'checkpoint_size_mb':os.path.getsize(a.checkpoint)/1024**2,
            'process_rss_mb':process.memory_info().rss/1024**2,'peak_cpu_rss_mb':peak_cpu_rss_mb,
            'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else None,
            'latency_protocol':f'one warm-up then median of {a.repeats} runs per sample/method; includes pipeline inference',
            'protocol':'Same held-out samples and GT validity mask. Low-res input is validity-weighted area reduction. No test-set tuning.',
            'summary':summary,'per_sample':rows,'ablation_summary':ablation_summary,'ablation_per_sample':ablations}
    os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True)
    with open(a.out,'w') as f: json.dump(report,f,indent=2)
    print(json.dumps(summary,indent=2)); print(json.dumps(ablation_summary,indent=2)); print(f'Wrote {a.out}')
if __name__=='__main__': main()
