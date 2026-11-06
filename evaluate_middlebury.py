"""Evaluate classical and hybrid depth upsampling on calibrated Middlebury scenes (metres)."""
import argparse, glob, os, json, time, statistics
from pathlib import Path
import numpy as np
import cv2
from src.pipeline import reconstruct
from src.metrics import compute_metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/processed/middlebury')
    p.add_argument('--out', default='results/middlebury_evaluation.json')
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--scale', type=int, default=4)
    p.add_argument('--repeats', type=int, default=2)
    a = p.parse_args()
    
    root = Path(a.data)
    rgb_files = {p.stem: p for p in (root / 'rgb').glob('*.npy')}
    gt_files = {p.stem: p for p in (root / 'gt').glob('*.npy')}
    scenes = sorted(set(rgb_files.keys()) & set(gt_files.keys()))
    if a.limit:
        scenes = scenes[:a.limit]
    if not scenes:
        raise SystemExit(f'No matching Middlebury RGB/GT pairs found in {root}')
    
    methods = [
        ('Nearest', 'Interpolation', 'nearest', 'guided'),
        ('Bilinear', 'Interpolation', 'bilinear', 'guided'),
        ('Bicubic', 'Interpolation', 'bicubic', 'guided'),
        ('KNN', 'KNN', 'bilinear', 'guided'),
        ('Guided filter', 'Classical', 'bilinear', 'guided'),
        ('Joint bilateral', 'Classical', 'bilinear', 'joint_bilateral'),
    ]
    
    rows = []
    print(f"Evaluating {len(scenes)} Middlebury scenes at {a.scale}x scale...")
    for idx, sid in enumerate(scenes, 1):
        rgb_orig = np.load(rgb_files[sid])
        truth_orig = np.load(gt_files[sid]).astype(np.float32)
        
        # Resize to 540x960 for reasonable evaluation speed on 1080p stereo
        h_t, w_t = 540, 960
        rgb = cv2.resize(rgb_orig, (w_t, h_t), interpolation=cv2.INTER_AREA)
        valid_orig = np.isfinite(truth_orig) & (truth_orig > 0)
        num = cv2.resize(np.where(valid_orig, truth_orig, 0).astype(np.float32), (w_t, h_t), interpolation=cv2.INTER_AREA)
        den = cv2.resize(valid_orig.astype(np.float32), (w_t, h_t), interpolation=cv2.INTER_AREA)
        truth = num / np.maximum(den, 1e-6)
        valid = den >= 0.75
        truth[~valid] = 0.0
        
        valid_depths = truth[valid]
        if not len(valid_depths): continue
        min_depth = max(0.1, float(valid_depths.min()))
        max_depth = float(np.percentile(valid_depths, 98))
        
        lh, lw = h_t // a.scale, w_t // a.scale
        den_lr = cv2.resize(valid.astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
        num_lr = cv2.resize(np.where(valid, truth, 0).astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
        low = num_lr / np.maximum(den_lr, 1e-6)
        low[den_lr < 0.75] = 0.0
        
        for label, mode, interp, filt in methods:
            times = []
            result = None
            for rep in range(a.repeats + 1):
                t = time.perf_counter()
                result = reconstruct(
                    rgb, low, a.scale, filt, None, mode,
                    interpolation=interp, min_depth=min_depth, max_depth=max_depth,
                    include_comparisons=False, depth_input_type='low_resolution'
                )
                elapsed = (time.perf_counter() - t) * 1000
                if rep > 0: times.append(elapsed)
            
            pred = min_depth + result['output'] * (max_depth - min_depth)
            pred[~result['output_valid']] = 0.0
            metrics = compute_metrics(pred, truth, valid, min_depth, max_depth)
            rows.append({
                'scene': sid,
                'scale': a.scale,
                'method': label,
                'mode': mode,
                **metrics,
                'median_latency_ms': float(statistics.median(times)) if times else 0.0
            })
        print(f"  Processed {idx}/{len(scenes)}: {sid}")

    keys = ['RMSE_m', 'MAE_m', 'PSNR_dB', 'SSIM', 'Edge_F1', 'Boundary_F1_2px', 'Boundary_MAE_m', 'prediction_coverage', 'median_latency_ms']
    summary = []
    for label, _, _, _ in methods:
        subset = [r for r in rows if r['method'] == label]
        summary.append({
            'method': label,
            'scale': a.scale,
            'n_scenes': len(subset),
            **{k: float(np.mean([r[k] for r in subset if r.get(k) is not None])) if any(r.get(k) is not None for r in subset) else None for k in keys}
        })
    
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump({
            'dataset': 'Middlebury Stereo Calibrated Metric Depth',
            'scale': a.scale,
            'n_scenes': len(scenes),
            'note': 'Metric depth converted from disparity PFM using fx, baseline and doffs from calib.txt.',
            'summary': summary,
            'per_scene': rows
        }, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote Middlebury evaluation results to {a.out}")

if __name__ == '__main__':
    main()
