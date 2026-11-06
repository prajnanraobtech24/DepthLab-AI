"""Inspect/convert NYU Depth V2 labelled MAT files to NPY train/val/test pairs.

The converter supports common MATLAB v7.3/HDF5 and classic MAT files. It prints
array shapes first and refuses unknown/ambiguous layouts rather than silently
transposing images or pairing samples incorrectly.
"""
from pathlib import Path
import argparse, json
import numpy as np


def load_mat(path):
    try:
        import h5py
        if h5py.is_hdf5(path):
            f=h5py.File(path,'r')
            keys=list(f.keys())
            return f, {k:f[k] for k in keys if hasattr(f[k],'shape')}
    except Exception: pass
    try:
        from scipy.io import loadmat
        d=loadmat(path)
        return None,{k:v for k,v in d.items() if not k.startswith('__') and hasattr(v,'shape')}
    except Exception as e: raise RuntimeError(f"Could not read MAT file as HDF5 or classic MAT: {e}")


def describe(arrays):
    for k,v in arrays.items(): print(f"{k}: shape={tuple(v.shape)} dtype={getattr(v,'dtype',type(v))}")


def choose_sample_axis(shape, channel=False):
    candidates=[i for i,n in enumerate(shape) if n not in ((3,) if channel else ()) and n not in (480,640)]
    if not candidates: raise ValueError(f"Cannot infer sample axis from shape {shape}")
    # The sample axis should be the largest remaining dimension (usually 1449).
    return max(candidates,key=lambda i:shape[i])


def get_rgb_sample(ds, idx):
    shape=tuple(ds.shape)
    if len(shape)!=4: raise ValueError(f"RGB array must be 4D, got {shape}")
    sample_axis=choose_sample_axis(shape,channel=True)
    sel=[slice(None)]*len(shape); sel[sample_axis]=idx; a=np.asarray(ds[tuple(sel)])
    if a.ndim!=3: raise ValueError(f"RGB sample is not 3D: {a.shape}")
    caxis=next((i for i,n in enumerate(a.shape) if n==3),None)
    if caxis is None: raise ValueError(f"Cannot identify RGB channel dimension in {a.shape}")
    a=np.moveaxis(a,caxis,-1)
    if a.shape[:2]==(640,480): a=np.transpose(a,(1,0,2))
    if a.shape[:2]!=(480,640): raise ValueError(f"Unexpected RGB spatial shape {a.shape}; inspect source orientation")
    if a.dtype!=np.uint8:
        a=a.astype(np.float32)
        if np.nanmax(a)<=1.0: a*=255.0
        a=np.clip(a,0,255).astype(np.uint8)
    return a


def get_depth_sample(ds,idx):
    shape=tuple(ds.shape)
    if len(shape)!=3: raise ValueError(f"Depth array must be 3D, got {shape}")
    sample_axis=choose_sample_axis(shape)
    sel=[slice(None)]*len(shape); sel[sample_axis]=idx; a=np.asarray(ds[tuple(sel)],np.float32).squeeze()
    if a.shape==(640,480): a=a.T
    if a.shape!=(480,640): raise ValueError(f"Unexpected depth spatial shape {a.shape}; inspect source orientation")
    a[~np.isfinite(a)]=0
    a[a<0]=0
    return a


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mat',required=True)
    ap.add_argument('--out',default='data/nyu')
    ap.add_argument('--inspect-only',action='store_true')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--train-fraction',type=float,default=.70)
    ap.add_argument('--val-fraction',type=float,default=.15)
    args=ap.parse_args()
    handle,arr=load_mat(args.mat)
    try:
        describe(arr)
        if args.inspect_only: return
        if 'images' not in arr or 'depths' not in arr:
            raise SystemExit(f"Expected 'images' and 'depths'. Found keys: {list(arr)}")
        if args.train_fraction<=0 or args.val_fraction<=0 or args.train_fraction+args.val_fraction>=1:
            raise SystemExit('Train and validation fractions must be positive and sum to less than one.')
        image_ds,depth_ds=arr['images'],arr['depths']
        n_img=image_ds.shape[choose_sample_axis(image_ds.shape,True)]
        n_dep=depth_ds.shape[choose_sample_axis(depth_ds.shape)]
        if n_img!=n_dep: raise SystemExit(f"Image/depth sample counts differ: {n_img} vs {n_dep}")
        n=n_img
        rng=np.random.default_rng(args.seed); order=rng.permutation(n)
        ntr=int(n*args.train_fraction); nva=int(n*args.val_fraction)
        splits={'train':order[:ntr],'val':order[ntr:ntr+nva],'test':order[ntr+nva:]}
        root=Path(args.out)
        for split,ids in splits.items():
            for sub in ('rgb','gt'): (root/split/sub).mkdir(parents=True,exist_ok=True)
            for idx in ids:
                rgb=get_rgb_sample(image_ds,int(idx)); depth=get_depth_sample(depth_ds,int(idx))
                if rgb.shape[:2]!=depth.shape: raise SystemExit(f"RGB/depth mismatch at sample {idx}: {rgb.shape} vs {depth.shape}")
                np.save(root/split/'rgb'/f'nyu_{idx:04d}.npy',rgb)
                np.save(root/split/'gt'/f'nyu_{idx:04d}.npy',depth)
            print(f"{split}: {len(ids)} pairs")
        meta={'source':str(Path(args.mat).name),'samples':n,'seed':args.seed,'split':'deterministic image-level 70/15/15','scene_disjoint':False,'depth_units':'metres (verify against source metadata)','rgb_shape':'480x640x3','depth_shape':'480x640'}
        (root/'split_metadata.json').write_text(json.dumps(meta,indent=2))
        print(f"Prepared {n} pairs in {root.resolve()}")
        print('IMPORTANT: this is an image-level split; it is not guaranteed scene-disjoint. Disclose this in the report.')
    finally:
        if handle is not None: handle.close()
if __name__=='__main__': main()
