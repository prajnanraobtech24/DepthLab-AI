"""Depth metrics with explicit masks and physical units."""
import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _edges(depth, valid):
    d = np.asarray(depth, np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    vals = mag[valid & np.isfinite(mag)]
    if vals.size == 0: return np.zeros(d.shape, bool)
    threshold = max(float(np.percentile(vals, 90)), 1e-6)
    return (mag >= threshold) & valid


def compute_metrics(pred_m, gt_m, valid_mask=None, min_depth=0.1, max_depth=10.0, boundary_tolerance_px=2):
    """MAE/RMSE in metres; PSNR/SSIM use one fixed metric depth range."""
    p = np.asarray(pred_m, np.float32); g = np.asarray(gt_m, np.float32)
    if p.shape != g.shape: raise ValueError("Prediction and ground truth must have identical dimensions")
    gt_valid = np.isfinite(g) & (g >= min_depth) & (g <= max_depth) & (g > 0)
    prediction_valid = np.isfinite(p) & (p > 0)
    valid = gt_valid & prediction_valid
    if valid_mask is not None: valid &= np.asarray(valid_mask, bool)
    if valid.sum() < 2:
        return {"RMSE_m":None,"MAE_m":None,"PSNR_dB":None,"SSIM":None,"Edge_F1":None,
                "Boundary_F1_2px":None,"Boundary_MAE_m":None,"NonBoundary_MAE_m":None,
                "valid_pixels":int(valid.sum()),"valid_fraction":float(valid.mean()),
                "prediction_coverage":float(valid.sum()/max(int(gt_valid.sum()),1))}
    p = np.clip(p, min_depth, max_depth); g = np.clip(g, min_depth, max_depth)
    err = p[valid] - g[valid]
    mae = float(np.mean(np.abs(err))); mse = float(np.mean(err**2)); rmse = float(np.sqrt(mse))
    pn = (p-min_depth)/(max_depth-min_depth); gn = (g-min_depth)/(max_depth-min_depth)
    psnr = float(10*np.log10(1.0/max(float(np.mean((pn[valid]-gn[valid])**2)), 1e-12)))
    try:
        _, smap = structural_similarity(gn, pn, data_range=1.0, full=True)
        # Only score SSIM windows whose full 7x7 support is valid.
        ssim_valid = cv2.erode(valid.astype(np.uint8), np.ones((7,7),np.uint8)) > 0
        ssim = float(np.mean(smap[ssim_valid])) if ssim_valid.any() else float(np.mean(smap[valid]))
    except Exception: ssim = None
    pe, ge = _edges(pn, valid), _edges(gn, valid)
    edge_f1 = float(2*(pe&ge).sum()/(pe.sum()+ge.sum()+1e-8)) if pe.any() or ge.any() else 1.0
    if pe.any() and ge.any():
        dgt = cv2.distanceTransform((~ge).astype(np.uint8), cv2.DIST_L2, 3)
        dp = cv2.distanceTransform((~pe).astype(np.uint8), cv2.DIST_L2, 3)
        precision = float(np.mean(dgt[pe] <= boundary_tolerance_px))
        recall = float(np.mean(dp[ge] <= boundary_tolerance_px))
        bf1 = 2*precision*recall/max(precision+recall, 1e-8)
    elif not pe.any() and not ge.any(): bf1 = 1.0
    else: bf1 = 0.0
    boundary = _edges(g, valid)
    # Evaluate a small band around GT depth discontinuities, not only edge pixels.
    band = cv2.dilate(boundary.astype(np.uint8), np.ones((5,5), np.uint8)) > 0
    band &= valid
    nonband = valid & ~band
    bmae = float(np.mean(np.abs(p[band]-g[band]))) if band.any() else None
    nbmae = float(np.mean(np.abs(p[nonband]-g[nonband]))) if nonband.any() else None
    return {"RMSE_m":rmse, "MAE_m":mae, "PSNR_dB":psnr, "SSIM":ssim,
            "Edge_F1":edge_f1, "Boundary_F1_2px":float(bf1),
            "Boundary_MAE_m":bmae, "NonBoundary_MAE_m":nbmae,
            "valid_pixels":int(valid.sum()), "valid_fraction":float(valid.mean()),
            "prediction_coverage":float(valid.sum()/max(int(gt_valid.sum()),1))}

# Compatibility helper names used by the optional diagnostic scripts.
def boundary_f1(pred, gt, valid_mask=None, tolerance_px=2):
    p=np.asarray(pred,np.float32); g=np.asarray(gt,np.float32)
    valid=np.ones(g.shape,bool) if valid_mask is None else np.asarray(valid_mask,bool)
    pe=_edges(p,valid); ge=_edges(g,valid)
    if not pe.any() and not ge.any(): return 1.0
    if not pe.any() or not ge.any(): return 0.0
    dgt=cv2.distanceTransform((~ge).astype(np.uint8),cv2.DIST_L2,3)
    dp=cv2.distanceTransform((~pe).astype(np.uint8),cv2.DIST_L2,3)
    pr=float(np.mean(dgt[pe]<=tolerance_px)); re=float(np.mean(dp[ge]<=tolerance_px))
    return float(2*pr*re/max(pr+re,1e-8))

def edge_preservation(pred, gt, valid_mask=None):
    return boundary_f1(pred,gt,valid_mask,tolerance_px=0)
