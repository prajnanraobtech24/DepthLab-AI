"""Generate 50 rich gallery benchmark scenes with RGB, depth, GT, and dual-split thumbnails."""
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
    # 1. 25 Curated NYU Indoor Scenes
    # -------------------------------------------------------------
    nyu_files = sorted([f.name for f in Path("data/processed/nyu/val/rgb").glob("*.npy")])[:25]
    nyu_titles = [
        "Living Room Lounge", "Kitchen Countertops", "Bedroom & Wardrobe", "Home Office Desk",
        "Bookshelf Study Nook", "Bathroom Vanity", "Dining Room Setting", "Classroom Auditorium",
        "Cafe Lounge Seating", "Modern Kitchen Island", "Library Book Stacks", "Conference Meeting Hall",
        "Study Bedroom Corner", "Computer Workstation", "Reception Counter", "Hallway Corridor",
        "Basement Utility Room", "Studio Apartment Corner", "Dining Table Geometry", "Kitchen Storage Cabinets",
        "Living Room Sofa View", "Office Filing Cabinet", "Restroom Sink & Mirror", "Master Bedroom Suite",
        "Foyer & Entry Hallway"
    ]
    
    print(f"Generating {len(nyu_files)} NYU Indoor Scenes...")
    for idx, fname in enumerate(nyu_files):
        sid = f"nyu_{idx+1:02d}"
        title = nyu_titles[idx] if idx < len(nyu_titles) else f"NYU Indoor Scene {idx+1}"
        rgb_p = Path("data/processed/nyu/val/rgb") / fname
        gt_p = Path("data/processed/nyu/val/gt") / fname
        if not (rgb_p.exists() and gt_p.exists()):
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
    # 2. 15 Curated KITTI Outdoor LiDAR Scenes
    # -------------------------------------------------------------
    kitti_files = sorted([f.name for f in Path("data/processed/kitti/val/rgb").glob("*.npy")])[:15]
    kitti_titles = [
        "Urban Avenue & Vehicles", "Highway Cruising Horizon", "Suburban Roadway & Trees",
        "Following Car Silhouette", "City Center Crossroad", "Residential Boulevard",
        "Cyclist & Sidewalk Zone", "Multi-Lane Thoroughfare", "Expressway Overpass",
        "Parked Vehicle Row", "Tree-Lined Country Road", "Commercial Storefront St.",
        "Intersection Turning Bay", "Suburban Street Crossing", "Two-Way Traffic Corridor"
    ]
    
    print(f"Generating {len(kitti_files)} KITTI Outdoor LiDAR Scenes...")
    for idx, fname in enumerate(kitti_files):
        sid = f"kitti_{idx+1:02d}"
        title = kitti_titles[idx] if idx < len(kitti_titles) else f"KITTI Scene {idx+1}"
        rgb_p = Path("data/processed/kitti/val/rgb") / fname
        sp_p = Path("data/processed/kitti/val/sparse") / fname
        gt_p = Path("data/processed/kitti/val/gt") / fname
        if not (rgb_p.exists() and sp_p.exists() and gt_p.exists()):
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
    # 3. 10 Curated Middlebury Stereo Geometric Scenes
    # -------------------------------------------------------------
    mid_files = sorted([f.name for f in Path("data/processed/middlebury/rgb").glob("*.npy")])[:10]
    mid_titles = [
        "Artroom Classical Statues", "Artroom Perspective Shift", "Bandsaw Heavy Workshop",
        "Bandsaw Blade Close-Up", "Chess Pieces Macro Edge", "Chess Board High-Angle",
        "Chess Endgame Setup", "Antique Curule Structure", "Curule Carved Armrest",
        "Curule Leg Geometry"
    ]
    
    print(f"Generating {len(mid_files)} Middlebury Stereo Scenes...")
    for idx, fname in enumerate(mid_files):
        sid = f"mid_{idx+1:02d}"
        title = mid_titles[idx] if idx < len(mid_titles) else f"Middlebury Scene {idx+1}"
        rgb_p = Path("data/processed/middlebury/rgb") / fname
        gt_p = Path("data/processed/middlebury/gt") / fname
        if not (rgb_p.exists() and gt_p.exists()):
            continue
        
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
    with open(out_json, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Successfully generated {len(manifest)} gallery benchmark scenes to {out_json}!")

if __name__ == "__main__":
    main()
