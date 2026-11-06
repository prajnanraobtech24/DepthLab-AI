"""Inventory Middlebury scenes, parse camera calibration, and convert disparity to metric depth."""
from pathlib import Path
import argparse, csv
import numpy as np
from PIL import Image

def read_pfm(path):
    with open(path, 'rb') as f:
        header = f.readline().decode('ascii').strip()
        if header not in ('PF', 'Pf'):
            raise ValueError(f'Not a PFM file: {path}')
        dims = f.readline().decode('ascii').strip()
        while dims.startswith('#'):
            dims = f.readline().decode('ascii').strip()
        w, h = map(int, dims.split())
        scale = float(f.readline().decode('ascii').strip())
        endian = '<' if scale < 0 else '>'
        channels = 3 if header == 'PF' else 1
        data = np.fromfile(f, endian + 'f')
        shape = (h, w, channels) if channels == 3 else (h, w)
        data = data.reshape(shape)
        data = np.flipud(data).copy()
        return data, abs(scale)

def parse_calib(path):
    cal = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.strip()
                if k in ('cam0', 'cam1'):
                    parts = v.strip('[]').split(';')
                    row0 = [float(x) for x in parts[0].split()]
                    cal[k + '_fx'] = row0[0]
                    cal[k + '_cx'] = row0[2]
                elif k in ('doffs', 'baseline', 'width', 'height', 'ndisp', 'vmin', 'vmax'):
                    cal[k] = float(v)
    return cal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/raw/middlebury')
    ap.add_argument('--out', default='data/processed/middlebury_manifest.csv')
    ap.add_argument('--convert', action='store_true', help='Convert disparity to metric depth NPY files')
    ap.add_argument('--save-dir', default='data/processed/middlebury')
    a = ap.parse_args()
    
    root = Path(a.root)
    if not root.exists():
        raise SystemExit(f'Middlebury raw directory not found: {root.resolve()}')
    
    save_dir = Path(a.save_dir)
    rgb_dir = save_dir / 'rgb'
    gt_dir = save_dir / 'gt'
    if a.convert:
        rgb_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        
    rows = []
    print(f"Scanning Middlebury directory: {root}")
    for scene in sorted(p for p in root.rglob('*') if p.is_dir()):
        calib_file = scene / 'calib.txt'
        pfm_file = scene / 'disp0.pfm'
        im0_file = scene / 'im0.png'
        if calib_file.exists() and pfm_file.exists() and im0_file.exists():
            cal = parse_calib(calib_file)
            disp, scale = read_pfm(pfm_file)
            fx = cal.get('cam0_fx', 1750.0)
            baseline_m = cal.get('baseline', 500.0) / 1000.0
            doffs = cal.get('doffs', 0.0)
            
            valid_disp = np.isfinite(disp) & (disp > 0)
            denom = disp + doffs
            valid = valid_disp & (denom > 1e-4)
            
            depth_m = np.zeros_like(disp, dtype=np.float32)
            depth_m[valid] = (fx * baseline_m) / denom[valid]
            
            valid_depths = depth_m[depth_m > 0]
            min_d = float(valid_depths.min()) if len(valid_depths) else 0.0
            max_d = float(valid_depths.max()) if len(valid_depths) else 0.0
            valid_pct = float(np.mean(depth_m > 0)) * 100.0
            
            scene_name = scene.name
            rows.append([scene_name, str(scene), fx, cal.get('baseline', 0.0), doffs, round(min_d, 3), round(max_d, 3), round(valid_pct, 1)])
            
            if a.convert:
                rgb = np.asarray(Image.open(im0_file).convert('RGB'))
                np.save(rgb_dir / f'{scene_name}.npy', rgb)
                np.save(gt_dir / f'{scene_name}.npy', depth_m)
            
            print(f"  Scene {scene_name:12s}: fx={fx:.1f}, B={cal.get('baseline', 0.0):.1f}mm, Depth=[{min_d:.2f}m - {max_d:.2f}m], Valid={valid_pct:.1f}%")

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene', 'path', 'fx', 'baseline_mm', 'doffs', 'min_depth_m', 'max_depth_m', 'valid_percent'])
        w.writerows(rows)
    print(f"\nFound and processed {len(rows)} Middlebury scenes. Manifest written to {out_path}.")
    if a.convert:
        print(f"Saved metric depth and RGB NPY pairs to {save_dir}")

if __name__ == '__main__':
    main()

