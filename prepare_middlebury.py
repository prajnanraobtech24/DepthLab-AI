"""Inventory Middlebury scenes and read PFM disparity maps."""
from pathlib import Path
import argparse, csv, re
import numpy as np

def read_pfm(path):
    with open(path,'rb') as f:
        header=f.readline().decode('ascii').strip()
        if header not in ('PF','Pf'): raise ValueError(f'Not a PFM file: {path}')
        dims=f.readline().decode('ascii').strip()
        while dims.startswith('#'): dims=f.readline().decode('ascii').strip()
        w,h=map(int,dims.split()); scale=float(f.readline().decode('ascii').strip())
        endian='<' if scale<0 else '>'; channels=3 if header=='PF' else 1
        data=np.fromfile(f,endian+'f')
        shape=(h,w,channels) if channels==3 else (h,w)
        data=data.reshape(shape); data=np.flipud(data).copy()
        return data,abs(scale)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='data/raw/middlebury'); ap.add_argument('--out',default='data/processed/middlebury_manifest.csv'); a=ap.parse_args()
    root=Path(a.root)
    if not root.exists(): raise SystemExit(f'Middlebury raw directory not found: {root.resolve()}')
    rows=[]
    for scene in sorted(p for p in root.rglob('*') if p.is_dir()):
        pfms=list(scene.glob('*.pfm'))
        if pfms:
            rows.append([str(scene),*[str(x) for x in pfms]])
            for pfm in pfms:
                try:
                    arr,scale=read_pfm(pfm); print(f'{pfm}: shape={arr.shape}, scale={scale}')
                except Exception as e: print(f'WARNING {pfm}: {e}')
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['scene','pfm_files']); w.writerows(rows)
    print(f'Found {len(rows)} scene directories with PFM files. Wrote {out}.')
    print('Disparity-to-depth conversion is not performed here: read each scene calibration and disparity convention first.')
if __name__=='__main__': main()
