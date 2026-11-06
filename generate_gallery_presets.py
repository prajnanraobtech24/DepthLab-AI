"""Generate 24 rich gallery scenes with RGB, depth, GT, and dual-split thumbnails."""
import os, json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def colorize_depth(depth_m, min_d, max_d, cmap=cv2.COLORMAP_TURBO):
    norm = np.clip((depth_m - min_d) / (max_d - min_d), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cmap)
    # Mask invalid zeros to dark navy/black
    invalid = ~np.isfinite(depth_m) | (depth_m <= 0)
    colored[invalid] = [12, 16, 24]
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

def make_split_thumb(rgb, depth_m, min_d, max_d, target_size=(320, 200)):
    h, w = rgb.shape[:2]
    colored = colorize_depth(depth_m, min_d, max_d)
    
    # Half RGB, half colored depth
    half_w = w // 2
    canvas = np.zeros_like(rgb)
    canvas[:, :half_w] = rgb[:, :half_w]
    canvas[:, half_w:] = colored[:, half_w:]
    
    # Add a glowing divider line at the boundary
    canvas[:, half_w-1:half_w+2] = [126, 224, 212]
    
    # Resize to thumbnail
    thumb = cv2.resize(canvas, target_size, interpolation=cv2.INTER_AREA)
    return thumb

def main():
    presets_dir = Path("web/presets")
    presets_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    
    # -------------------------------------------------------------
    # 1. 10 Curated NYU Indoor Scenes
    # -------------------------------------------------------------
    nyu_scenes = [
        ("nyu_01", "Living Room Lounge", "nyu_0003.npy"),
        ("nyu_02", "Kitchen Countertops", "nyu_0016.npy"),
        ("nyu_03", "Bedroom & Wardrobe", "nyu_0019.npy"),
        ("nyu_04", "Home Office Desk", "nyu_0052.npy"),
        ("nyu_05", "Bookshelf Study Nook", "nyu_0054.npy"),
        ("nyu_06", "Bathroom Vanity", "nyu_0058.npy"),
        ("nyu_07", "Dining Room Setting", "nyu_0073.npy"),
        ("nyu_08", "Classroom Auditorium", "nyu_0084.npy"),
        ("nyu_09", "Cafe Lounge Seating", "nyu_0089.npy"),
        ("nyu_10", "Modern Kitchen Island", "nyu_0099.npy"),
    ]
    
    print("Generating 10 NYU Indoor Scenes...")
    for sid, title, fname in nyu_scenes:
        rgb_p = Path("data/processed/nyu/val/rgb") / fname
        gt_p = Path("data/processed/nyu/val/gt") / fname
        if not (rgb_p.exists() and gt_p.exists()):
            print(f"Skipping {fname}, file missing")
            continue
        
        rgb = np.load(rgb_p)
        gt = np.load(gt_p).astype(np.float32)
        h, w = gt.shape
        valid = np.isfinite(gt) & (gt >= 0.1) & (gt <= 10.0) & (gt > 0)
        
        # 4x downsample
        lh, lw = h // 4, w // 4
        num = cv2.resize(np.where(valid, gt, 0).astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
        den = cv2.resize(valid.astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
        low = num / np.maximum(den, 1e-6)
        low[den < 0.75] = 0.0
        
        # Save files
        Image.fromarray(rgb).save(presets_dir / f"{sid}_rgb.png")
        Image.fromarray((np.clip(low * 1000.0, 0, 65535)).astype(np.uint16)).save(presets_dir / f"{sid}_depth.png")
        Image.fromarray((np.clip(gt * 1000.0, 0, 65535)).astype(np.uint16)).save(presets_dir / f"{sid}_gt.png")
        
        thumb = make_split_thumb(rgb, gt, 0.1, 10.0)
        Image.fromarray(thumb).save(presets_dir / f"{sid}_thumb.png")
        
        manifest.append({
            "id": sid,
            "title": title,
            "dataset": "NYU Depth V2",
            "category": "indoor",
            "badge": "INDOOR · 4×",
            "badgeClass": "indoor",
            "resolution": f"{w} × {h}",
            "range": "0.1 – 10.0 m",
            "profile": "NYU indoor · 10 m",
            "units": "millimetres",
            "inputType": "low_resolution",
            "mode": "Hybrid",
            "method": "guided",
            "scale": 4,
            "rgbUrl": f"/assets/presets/{sid}_rgb.png",
            "depthUrl": f"/assets/presets/{sid}_depth.png",
            "gtUrl": f"/assets/presets/{sid}_gt.png",
            "thumbUrl": f"/assets/presets/{sid}_thumb.png"
        })
    
    # -------------------------------------------------------------
    # 2. 8 Curated KITTI Automotive LiDAR Scenes
    # -------------------------------------------------------------
    kitti_scenes = [
        ("kitti_01", "Urban Avenue & Vehicles", "kitti_000000.npy"),
        ("kitti_02", "Highway Cruising Horizon", "kitti_000001.npy"),
        ("kitti_03", "Suburban Roadway & Trees", "kitti_000002.npy"),
        ("kitti_04", "Following Car Silhouette", "kitti_000003.npy"),
        ("kitti_05", "City Center Crossroad", "kitti_000004.npy"),
        ("kitti_06", "Residential Boulevard", "kitti_000005.npy"),
        ("kitti_07", "Cyclist & Sidewalk Zone", "kitti_000006.npy"),
        ("kitti_08", "Multi-Lane Thoroughfare", "kitti_000007.npy"),
    ]
    
    print("Generating 8 KITTI Outdoor LiDAR Scenes...")
    for sid, title, fname in kitti_scenes:
        rgb_p = Path("data/processed/kitti/val/rgb") / fname
        sp_p = Path("data/processed/kitti/val/sparse") / fname
        gt_p = Path("data/processed/kitti/val/gt") / fname
        if not (rgb_p.exists() and sp_p.exists() and gt_p.exists()):
            print(f"Skipping {fname}, file missing")
            continue
        
        rgb = np.load(rgb_p)
        sp = np.load(sp_p).astype(np.float32)
        gt = np.load(gt_p).astype(np.float32)
        h, w = gt.shape
        
        Image.fromarray(rgb).save(presets_dir / f"{sid}_rgb.png")
        Image.fromarray((np.clip(sp * 256.0, 0, 65535)).astype(np.uint16)).save(presets_dir / f"{sid}_depth.png")
        Image.fromarray((np.clip(gt * 256.0, 0, 65535)).astype(np.uint16)).save(presets_dir / f"{sid}_gt.png")
        
        thumb = make_split_thumb(rgb, gt, 0.1, 80.0)
        Image.fromarray(thumb).save(presets_dir / f"{sid}_thumb.png")
        
        manifest.append({
            "id": sid,
            "title": title,
            "dataset": "KITTI LiDAR",
            "category": "outdoor",
            "badge": "LIDAR · 80M",
            "badgeClass": "outdoor",
            "resolution": f"{w} × {h}",
            "range": "0.1 – 80.0 m",
            "profile": "KITTI outdoor · 80 m",
            "units": "kitti_256",
            "inputType": "full_resolution_sparse",
            "mode": "Classical",
            "method": "joint_bilateral",
            "scale": 4,
            "rgbUrl": f"/assets/presets/{sid}_rgb.png",
            "depthUrl": f"/assets/presets/{sid}_depth.png",
            "gtUrl": f"/assets/presets/{sid}_gt.png",
            "thumbUrl": f"/assets/presets/{sid}_thumb.png"
        })

    # -------------------------------------------------------------
    # 3. 6 Curated Middlebury Stereo Geometric Scenes
    # -------------------------------------------------------------
    mid_scenes = [
        ("mid_01", "Artroom Classical Statues", "artroom1.npy"),
        ("mid_02", "Bandsaw Heavy Workshop", "bandsaw1.npy"),
        ("mid_03", "Chess Pieces Macro Edge", "chess1.npy"),
        ("mid_04", "Antique Curule Structure", "curule1.npy"),
        ("mid_05", "Industrial Ladder Rungs", "ladder1.npy"),
        ("mid_06", "Pendulum Mechanism Study", "pendulum1.npy"),
    ]
    
    print("Generating 6 Middlebury Stereo Scenes...")
    for sid, title, fname in mid_scenes:
        rgb_p = Path("data/processed/middlebury/rgb") / fname
        gt_p = Path("data/processed/middlebury/gt") / fname
        if not (rgb_p.exists() and gt_p.exists()):
            print(f"Skipping {fname}, file missing")
            continue
        
        # Standardize size to 640x360
        rgb = cv2.resize(np.load(rgb_p), (640, 360), interpolation=cv2.INTER_AREA)
        gt_full = cv2.resize(np.load(gt_p), (640, 360), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        valid = gt_full > 0
        h, w = gt_full.shape
        lh, lw = h // 4, w // 4
        num = cv2.resize(np.where(valid, gt_full, 0).astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
        den = cv2.resize(valid.astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
        low = num / np.maximum(den, 1e-6)
        low[den < 0.75] = 0.0
        
        Image.fromarray(rgb).save(presets_dir / f"{sid}_rgb.png")
        Image.fromarray((np.clip(low * 1000.0, 0, 65535)).astype(np.uint16)).save(presets_dir / f"{sid}_depth.png")
        Image.fromarray((np.clip(gt_full * 1000.0, 0, 65535)).astype(np.uint16)).save(presets_dir / f"{sid}_gt.png")
        
        thumb = make_split_thumb(rgb, gt_full, 0.1, 10.0)
        Image.fromarray(thumb).save(presets_dir / f"{sid}_thumb.png")
        
        manifest.append({
            "id": sid,
            "title": title,
            "dataset": "Middlebury Stereo",
            "category": "stereo",
            "badge": "STEREO · 4×",
            "badgeClass": "stereo",
            "resolution": f"{w} × {h}",
            "range": "0.1 – 10.0 m",
            "profile": "NYU indoor · 10 m",
            "units": "millimetres",
            "inputType": "low_resolution",
            "mode": "Hybrid",
            "method": "guided",
            "scale": 4,
            "rgbUrl": f"/assets/presets/{sid}_rgb.png",
            "depthUrl": f"/assets/presets/{sid}_depth.png",
            "gtUrl": f"/assets/presets/{sid}_gt.png",
            "thumbUrl": f"/assets/presets/{sid}_thumb.png"
        })
    
    out_json = presets_dir / "gallery_manifest.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSuccessfully generated {len(manifest)} benchmark scenes in {presets_dir}!")
    print(f"Saved manifest to {out_json}")

if __name__ == "__main__":
    main()
