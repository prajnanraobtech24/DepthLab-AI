"""End-to-end RGB-guided depth reconstruction with explicit masks and safe fallbacks."""
from __future__ import annotations
import os, time
from functools import lru_cache
import cv2
import numpy as np
import torch
from .classical import masked_resize, classical_refine, normalize_fixed, knn_fill, fill_holes
from .model import DepthRefineNet

@lru_cache(maxsize=4)
def load_model(checkpoint, device):
    model = DepthRefineNet().to(device)
    if not checkpoint or not os.path.isfile(checkpoint): return model, False
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else: model.load_state_dict(state)
    model.eval()
    return model, True


def reconstruct(rgb, sparse, scale=4, classical="guided", checkpoint=None, mode="Hybrid", alpha=0.5,
                interpolation="bilinear", min_depth=0.1, max_depth=10.0, include_comparisons=True,
                use_rgb_guidance=True, use_valid_mask=True, depth_input_type="auto"):
    start = time.perf_counter()
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3: raise ValueError("RGB guidance must be HxWx3")
    h, w = rgb.shape[:2]
    raw = np.asarray(sparse, np.float32)
    if raw.ndim != 2: raise ValueError("Depth input must be a 2D array")
    if scale not in (2, 4, 8): raise ValueError("Scale must be 2, 4 or 8")
    if depth_input_type not in ("auto", "low_resolution", "full_resolution_sparse"):
        raise ValueError("Unsupported depth_input_type")
    is_full = raw.shape == (h, w) if depth_input_type == "auto" else depth_input_type == "full_resolution_sparse"
    if is_full and raw.shape != (h, w): raise ValueError("Full-resolution sparse depth must match RGB dimensions")
    if not is_full and (raw.shape[0] >= h or raw.shape[1] >= w): raise ValueError("Select full-resolution sparse mode when depth dimensions match RGB")
    if not is_full and (abs(raw.shape[0]*scale-h)>scale or abs(raw.shape[1]*scale-w)>scale):
        raise ValueError(f"Low-resolution depth dimensions {raw.shape} do not match the selected {scale}x scale for RGB {(h,w)}. Check the scale or input type.")
    valid_input = np.isfinite(raw) & (raw >= min_depth) & (raw <= max_depth) & (raw > 0)
    raw_norm, _ = normalize_fixed(raw, min_depth, max_depth)
    base, base_support = masked_resize(raw_norm, valid_input, (h, w), interpolation)
    # Nearest mask upsampling preserves the discrete observation support.
    mask_full = cv2.resize(valid_input.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not mask_full.any(): raise ValueError("Depth input contains no valid measurements in the selected range")
    # Fill only for filters/model input; validity remains separate.
    filled_base = fill_holes(base, mask_full)
    if is_full:
        filled_base[mask_full] = raw_norm[mask_full]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_exists = bool(checkpoint and os.path.isfile(checkpoint))
    need_classical = include_comparisons or mode in ("Classical", "Hybrid") or (mode == "Deep learning" and not checkpoint_exists)
    c = classical_refine(rgb, filled_base, method=classical, valid_mask=mask_full) if need_classical else filled_base.copy()
    # KNN operates on original sample grid then is resized to RGB dimensions.
    need_knn = include_comparisons or mode == "KNN"
    if need_knn:
        knn_input = knn_fill(rgb, raw_norm, valid_input, k=5)
        knn_full = cv2.resize(knn_input, (w, h), interpolation=cv2.INTER_LINEAR) if knn_input.shape != (h,w) else knn_input
        if is_full: knn_full[mask_full] = raw_norm[mask_full]
    else: knn_full = base.copy()
    need_dl = include_comparisons or mode in ("Deep learning", "Hybrid")
    trained = False; dl = c.copy()
    if need_dl and checkpoint_exists:
        model, trained = load_model(checkpoint, device)
        rgb_in = rgb.astype(np.float32) / 255.0 if use_rgb_guidance else np.zeros((h,w,3), np.float32)
        mask_in = mask_full.astype(np.float32) if use_valid_mask else np.zeros((h,w), np.float32)
        x = np.concatenate([rgb_in, filled_base[...,None], mask_in[...,None]], axis=2)
        ph, pw = (4-h%4)%4, (4-w%4)%4
        x = np.pad(x, ((0,ph),(0,pw),(0,0)), mode="reflect")
        with torch.inference_mode():
            t = torch.from_numpy(x.transpose(2,0,1)[None].copy()).float().to(device)
            dl = model(t)[0,0].cpu().numpy()[:h,:w]
        if is_full: dl[mask_full] = raw_norm[mask_full]
    # Adaptive fusion: confidence favours classical output near supported depth
    # discontinuities and DL elsewhere; this is a heuristic, to be ablated.
    if include_comparisons or mode == "Hybrid":
        boundary, grad = __import__('src.classical', fromlist=['depth_discontinuity']).depth_discontinuity(filled_base, np.ones((h,w),bool))
        confidence = np.clip(0.5 + 2.0 * grad, 0.0, 1.0)
        confidence = np.maximum(confidence, boundary.astype(np.float32) * 0.75)
        dl_weight = float(np.clip(alpha, 0.0, 1.0)) * (1.0 - 0.7*confidence)
        hybrid = dl_weight*dl + (1.0-dl_weight)*c if trained else c.copy()
    else: hybrid = c.copy()
    if is_full:
        hybrid[mask_full] = raw_norm[mask_full]; c[mask_full] = raw_norm[mask_full]
    if mode == "Interpolation": out = base; output_valid = base_support
    elif mode == "KNN": out = knn_full; output_valid = np.ones((h,w),bool)
    elif mode == "Classical": out = c; output_valid = np.ones((h,w),bool)
    elif mode == "Deep learning": out = dl if trained else c; output_valid = np.ones((h,w),bool)
    elif mode == "Hybrid": out = hybrid; output_valid = np.ones((h,w),bool)
    else: raise ValueError(f"Unknown mode: {mode}")
    return {"output":np.clip(out,0,1), "output_valid":output_valid, "interpolation":base, "knn":knn_full, "classical":c,
            "deep":dl, "hybrid":hybrid, "trained":trained, "valid_mask":mask_full,
            "latency_ms":(time.perf_counter()-start)*1000, "device":device,
            "input_kind":"full_resolution_sparse" if is_full else "low_resolution"}
