"""Extract and format demo preset assets for instant 1-click evaluation in UI."""
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

def main():
    presets_dir = Path("D:/DepthLab_AI_Final/depthlab_ai/web/presets")
    presets_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. NYU Sample
    nyu_rgb_p = Path("D:/DepthLab_AI_Final/depthlab_ai/data/processed/nyu/val/rgb/nyu_0003.npy")
    nyu_gt_p = Path("D:/DepthLab_AI_Final/depthlab_ai/data/processed/nyu/val/gt/nyu_0003.npy")
    if nyu_rgb_p.exists() and nyu_gt_p.exists():
        rgb = np.load(nyu_rgb_p)
        gt = np.load(nyu_gt_p).astype(np.float32)
        h, w = gt.shape
        valid = np.isfinite(gt) & (gt >= 0.1) & (gt <= 10.0) & (gt > 0)
        
        # 4x downsampling
        lh, lw = h // 4, w // 4
        num = cv2.resize(np.where(valid, gt, 0).astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
        den = cv2.resize(valid.astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
        low = num / np.maximum(den, 1e-6)
        low[den < 0.75] = 0.0
        
        # Save RGB as PNG
        Image.fromarray(rgb).save(presets_dir / "nyu_rgb.png")
        # Save low-res depth as uint16 millimetres PNG
        low_mm = np.clip(low * 1000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(low_mm).save(presets_dir / "nyu_depth.png")
        # Save GT depth as uint16 millimetres PNG
        gt_mm = np.clip(gt * 1000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(gt_mm).save(presets_dir / "nyu_gt.png")
        print("Generated NYU preset assets.")

    # 2. KITTI Sample
    kitti_rgb_p = Path("D:/DepthLab_AI_Final/depthlab_ai/data/processed/kitti/val/rgb/kitti_000000.npy")
    kitti_sp_p = Path("D:/DepthLab_AI_Final/depthlab_ai/data/processed/kitti/val/sparse/kitti_000000.npy")
    kitti_gt_p = Path("D:/DepthLab_AI_Final/depthlab_ai/data/processed/kitti/val/gt/kitti_000000.npy")
    if kitti_rgb_p.exists() and kitti_sp_p.exists() and kitti_gt_p.exists():
        rgb = np.load(kitti_rgb_p)
        sp = np.load(kitti_sp_p).astype(np.float32)
        gt = np.load(kitti_gt_p).astype(np.float32)
        
        # Save RGB
        Image.fromarray(rgb).save(presets_dir / "kitti_rgb.png")
        # Save sparse depth as uint16 with 256 units per metre (KITTI convention)
        sp_kitti = np.clip(sp * 256.0, 0, 65535).astype(np.uint16)
        Image.fromarray(sp_kitti).save(presets_dir / "kitti_depth.png")
        # Save GT as uint16
        gt_kitti = np.clip(gt * 256.0, 0, 65535).astype(np.uint16)
        Image.fromarray(gt_kitti).save(presets_dir / "kitti_gt.png")
        print("Generated KITTI preset assets.")

    # 3. Middlebury Sample
    mid_rgb_p = Path("D:/DepthLab_AI_Final/depthlab_ai/data/processed/middlebury/rgb/artroom1.npy")
    mid_gt_p = Path("D:/DepthLab_AI_Final/depthlab_ai/data/processed/middlebury/gt/artroom1.npy")
    if mid_rgb_p.exists() and mid_gt_p.exists():
        rgb = cv2.resize(np.load(mid_rgb_p), (640, 360), interpolation=cv2.INTER_AREA)
        gt_full = cv2.resize(np.load(mid_gt_p), (640, 360), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        valid = gt_full > 0
        h, w = gt_full.shape
        lh, lw = h // 4, w // 4
        num = cv2.resize(np.where(valid, gt_full, 0).astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
        den = cv2.resize(valid.astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
        low = num / np.maximum(den, 1e-6)
        low[den < 0.75] = 0.0
        
        Image.fromarray(rgb).save(presets_dir / "mid_rgb.png")
        low_mm = np.clip(low * 1000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(low_mm).save(presets_dir / "mid_depth.png")
        gt_mm = np.clip(gt_full * 1000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(gt_mm).save(presets_dir / "mid_gt.png")
        print("Generated Middlebury preset assets.")

if __name__ == '__main__':
    main()
