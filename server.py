"""FastAPI application for DepthLab AI."""
from __future__ import annotations
import base64, io, os
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.pipeline import reconstruct
from src.classical import depth_discontinuity
from src.metrics import compute_metrics

ROOT = Path(__file__).resolve().parent
CHECKPOINT = os.getenv("DEPTHLAB_CHECKPOINT", str(ROOT/"checkpoints"/"depthrefinenet.pt"))
app = FastAPI(title="DepthLab AI", version="2.0.0", description="Edge-preserving depth upsampling and refinement")
app.mount("/assets", StaticFiles(directory=ROOT/"web"), name="assets")

@app.get("/")
def home(): return FileResponse(ROOT/"web"/"index.html")

@app.get("/favicon.ico")
def favicon():
    fav = ROOT/"web"/"favicon.ico"
    if fav.is_file(): return FileResponse(fav)
    return FileResponse(ROOT/"web"/"index.html")

@app.get("/api/health")
def health():
    return {"status":"ok", "model_trained":Path(CHECKPOINT).is_file(), "checkpoint":Path(CHECKPOINT).name if Path(CHECKPOINT).is_file() else None}


def decode_rgb(data: bytes):
    try: return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except (UnidentifiedImageError, OSError): raise HTTPException(400, "Cannot read RGB image. Use PNG or JPEG.")


def decode_depth(data: bytes, filename: str):
    try:
        if filename.lower().endswith(".npy"):
            arr = np.load(io.BytesIO(data), allow_pickle=False)
        else:
            arr = np.asarray(Image.open(io.BytesIO(data)))
            if arr.ndim == 3: arr = arr[...,0]
        if arr.ndim != 2: raise ValueError("Depth must be a single-channel 2D array")
        arr = arr.astype(np.float32)
        if not np.isfinite(arr).any() or not np.any(np.isfinite(arr) & (arr > 0)):
            raise ValueError("Depth map has no finite positive measurements")
        return arr
    except Exception as exc: raise HTTPException(400, f"Cannot read depth map: {exc}")


def png_depth(depth_m: np.ndarray, units_per_metre: int):
    # uint16 millimetres for indoor ranges; KITTI 1/256 m encoding for ranges >65.535 m.
    a = np.asarray(depth_m, np.float32)
    out = np.zeros(a.shape, np.uint16)
    valid = np.isfinite(a) & (a > 0)
    out[valid] = np.clip(np.rint(a[valid]*units_per_metre), 1, 65535).astype(np.uint16)
    b = io.BytesIO(); Image.fromarray(out).save(b, format="PNG"); return b.getvalue()

@app.post("/api/reconstruct")
async def api_reconstruct(
    rgb_file: UploadFile=File(...), depth_file: UploadFile=File(...),
    scale: int=Form(4), method: str=Form("guided"), interpolation: str=Form("bilinear"),
    mode: str=Form("Hybrid"), alpha: float=Form(.5), gt_file: Optional[UploadFile]=File(None),
    depth_profile: str=Form("NYU indoor · 10 m"), depth_units: str=Form("metres"),
    depth_input_type: str=Form("auto"),
):
    if scale not in (2,4,8): raise HTTPException(400, "Scale must be 2x, 4x or 8x")
    if method not in ("guided","joint_bilateral"): raise HTTPException(400, "Unknown edge-aware filter")
    if interpolation not in ("nearest","bilinear","bicubic"): raise HTTPException(400, "Unknown interpolation")
    if mode not in ("Interpolation","KNN","Classical","Deep learning","Hybrid"): raise HTTPException(400, "Unknown reconstruction mode")
    if depth_input_type not in ("auto","low_resolution","full_resolution_sparse"): raise HTTPException(400, "Unknown depth input type")
    rb, db = await rgb_file.read(), await depth_file.read()
    if len(rb)>15*1024*1024 or len(db)>15*1024*1024: raise HTTPException(413,"Each input file must be under 15 MB")
    rgb, depth = decode_rgb(rb), decode_depth(db, depth_file.filename or "depth.png")
    h,w=rgb.shape[:2]
    if h<32 or w<32: raise HTTPException(400,"RGB image must be at least 32 × 32 pixels")
    if depth.shape[0]>h or depth.shape[1]>w: raise HTTPException(400,"Depth dimensions cannot exceed RGB dimensions")
    profiles={"NYU indoor · 10 m":(0.1,10.0),"KITTI outdoor · 80 m":(0.1,80.0)}
    if depth_profile not in profiles: raise HTTPException(400,"Select a supported depth profile")
    if depth_units not in ("metres","millimetres","kitti_256"): raise HTTPException(400,"Depth units must be metres, millimetres, or kitti_256")
    lo,hi=profiles[depth_profile]
    if depth_units=="millimetres": depth=depth/1000.0
    elif depth_units=="kitti_256": depth=depth/256.0
    elif depth.max() > hi:
        if depth.max() / 256.0 <= hi * 1.5: depth = depth / 256.0
        elif depth.max() / 1000.0 <= hi * 1.5: depth = depth / 1000.0
    try:
        r=reconstruct(rgb,depth,scale,method,CHECKPOINT,mode,alpha,interpolation,lo,hi,True,True,True,depth_input_type)
    except ValueError as exc: raise HTTPException(400,str(exc))
    def enc(x):
        # Render normalized depth with a fixed grayscale palette for the UI.
        img=np.uint8(np.clip(x,0,1)*255)
        b=io.BytesIO(); Image.fromarray(img).save(b,format="PNG")
        return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()
    output_m=lo+r["output"]*(hi-lo)
    output_m[~r["output_valid"]]=0.0
    npy=io.BytesIO(); np.save(npy,output_m.astype(np.float32))
    png_units_per_metre=256 if hi>65.535 else 1000
    png_encoding="KITTI_1_over_256_m" if png_units_per_metre==256 else "millimetres"
    boundary,_=depth_discontinuity(r["output"],np.ones((h,w),bool))
    payload={"interpolation":enc(r["interpolation"]),"knn":enc(r["knn"]),"classical":enc(r["classical"]),
             "deep":enc(r["deep"]),"hybrid":enc(r["hybrid"]),"output":enc(r["output"]),"boundary":enc(boundary.astype(np.float32)),
             "latency_ms":round(r["latency_ms"],2),"device":r["device"],"trained":r["trained"],
             "scale":scale,"input_size":[w,h],"depth_size":[depth.shape[1],depth.shape[0]],
             "valid_fraction":round(float(np.mean(np.isfinite(depth)&(depth>0))),4),
             "input_kind":r["input_kind"],"depth_range_m":[lo,hi],
             "output_png":"data:image/png;base64,"+base64.b64encode(png_depth(output_m,png_units_per_metre)).decode(),"png_encoding":png_encoding,"png_units_per_metre":png_units_per_metre,
             "output_npy":base64.b64encode(npy.getvalue()).decode()}
    if gt_file and gt_file.filename:
        gb=await gt_file.read()
        if len(gb)>15*1024*1024: raise HTTPException(413,"Ground-truth file must be under 15 MB")
        gt=decode_depth(gb,gt_file.filename)
        if depth_units=="millimetres": gt=gt/1000.0
        elif depth_units=="kitti_256": gt=gt/256.0
        elif gt.max() > hi:
            if gt.max() / 256.0 <= hi * 1.5: gt = gt / 256.0
            elif gt.max() / 1000.0 <= hi * 1.5: gt = gt / 1000.0
        if gt.shape!=(h,w): raise HTTPException(400,"Ground truth must match RGB dimensions exactly")
        payload["metrics"]=compute_metrics(output_m,gt,valid_mask=r["output_valid"],min_depth=lo,max_depth=hi)
        err=np.zeros_like(output_m,np.float32); good=np.isfinite(gt)&(gt>=lo)&(gt<=hi)&(gt>0)&r["output_valid"]
        err[good]=np.abs(output_m[good]-gt[good])/(hi-lo)
        heat=cv2.applyColorMap(np.uint8(np.clip(err,0,1)*255),cv2.COLORMAP_TURBO)
        heat=cv2.cvtColor(heat,cv2.COLOR_BGR2RGB); eb=io.BytesIO(); Image.fromarray(heat).save(eb,format="PNG")
        payload["error_map"]="data:image/png;base64,"+base64.b64encode(eb.getvalue()).decode()
        payload["metric_note"]=f"MAE/RMSE in metres; PSNR uses fixed {hi-lo:.1f} m range; invalid GT pixels excluded; prediction coverage is reported. Boundary F1 uses 2 px tolerance and a 5×5 GT boundary band for boundary MAE."
    return payload
