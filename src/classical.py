"""Classical depth upsampling/refinement methods for DepthLab AI.

All depth arrays entering the reconstruction pipeline are fixed-range normalized
[0, 1]. Invalid measurements are represented by a separate boolean mask.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.spatial import cKDTree

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


def normalize_fixed(depth, min_depth=0.1, max_depth=10.0):
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d >= min_depth) & (d <= max_depth) & (d > 0)
    out = np.zeros(d.shape, np.float32)
    out[valid] = (d[valid] - min_depth) / (max_depth - min_depth)
    return np.clip(out, 0, 1), valid


def masked_resize(depth, valid, size, interpolation="bilinear"):
    """Resize a depth map without blending invalid zeros into valid samples."""
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"Unknown interpolation: {interpolation}")
    h, w = size
    d = np.asarray(depth, np.float32)
    m = np.asarray(valid, bool) & np.isfinite(d)
    if d.ndim != 2 or m.shape != d.shape:
        raise ValueError("depth and validity mask must be matching 2D arrays")
    flag = INTERPOLATIONS[interpolation]
    numerator = cv2.resize(np.where(m, d, 0).astype(np.float32), (w, h), interpolation=flag)
    denominator = cv2.resize(m.astype(np.float32), (w, h), interpolation=flag)
    denominator = np.clip(denominator, 0.0, 1.0)
    out = numerator / np.maximum(denominator, 1e-6)
    out[denominator < 1e-5] = 0
    return np.clip(out, 0, 1), denominator > 1e-5


def resize_depth(depth, scale=4, method="bilinear", valid_mask=None):
    d = np.asarray(depth, np.float32)
    h, w = d.shape[:2]
    mask = (np.isfinite(d) & (d > 0)) if valid_mask is None else np.asarray(valid_mask, bool)
    return masked_resize(d, mask, (h * scale, w * scale), method)[0]


def fill_holes(depth, valid_mask=None, max_hole_area=None):
    """Nearest-valid fill; never modifies measured pixels. Optional large-hole guard."""
    d = np.asarray(depth, np.float32).copy()
    valid = np.isfinite(d) & (d > 0) if valid_mask is None else (np.asarray(valid_mask, bool) & np.isfinite(d))
    if valid.all() or not valid.any():
        return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    invalid = (~valid).astype(np.uint8)
    # OpenCV returns nearest valid pixel coordinates for each invalid pixel.
    from scipy.ndimage import distance_transform_edt
    _, indices = distance_transform_edt(~valid, return_indices=True)
    filled = d.copy()
    filled[~valid] = d[indices[0][~valid], indices[1][~valid]]
    if max_hole_area is not None:
        n, labels_cc, stats, _ = cv2.connectedComponentsWithStats(invalid, 8)
        for label in range(1, n):
            if stats[label, cv2.CC_STAT_AREA] > max_hole_area:
                filled[labels_cc == label] = 0
    filled[valid] = d[valid]
    return filled


def guided_filter(guide, src, radius=8, eps=1e-3):
    I = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 if guide.ndim == 3 else guide.astype(np.float32)
    if I.max(initial=0) > 1: I /= 255.0
    p = np.asarray(src, np.float32)
    k = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(I, -1, k, normalize=True)
    mean_p = cv2.boxFilter(p, -1, k, normalize=True)
    corr_I = cv2.boxFilter(I * I, -1, k, normalize=True)
    corr_Ip = cv2.boxFilter(I * p, -1, k, normalize=True)
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return np.clip(cv2.boxFilter(a, -1, k, normalize=True) * I + cv2.boxFilter(b, -1, k, normalize=True), 0, 1)


def joint_bilateral(guide, depth, diameter=7, sigma_color=0.08, sigma_space=5.0):
    """RGB-guided joint bilateral filter; ximgproc if available, vectorized fallback otherwise."""
    g = np.asarray(guide, np.float32)
    if g.max(initial=0) > 1: g /= 255.0
    p = np.asarray(depth, np.float32)
    if hasattr(cv2, "ximgproc"):
        try:
            return np.clip(cv2.ximgproc.jointBilateralFilter(g, p, diameter, sigma_color, sigma_space), 0, 1)
        except cv2.error:
            pass
    r = max(1, diameter // 2)
    gp = cv2.copyMakeBorder(g, r, r, r, r, cv2.BORDER_REFLECT)
    pp = cv2.copyMakeBorder(p, r, r, r, r, cv2.BORDER_REFLECT)
    h, w = p.shape
    center = gp[r:r+h, r:r+w]
    acc = np.zeros_like(p); weights = np.zeros_like(p)
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            shifted_g = gp[r+dy:r+dy+h, r+dx:r+dx+w]
            shifted_p = pp[r+dy:r+dy+h, r+dx:r+dx+w]
            dc = np.sum((center-shifted_g)**2, axis=2)
            ds = dx*dx + dy*dy
            wt = np.exp(-dc/(2*sigma_color*sigma_color+1e-8)-ds/(2*sigma_space*sigma_space+1e-8)).astype(np.float32)
            acc += wt*shifted_p; weights += wt
    return np.clip(acc/np.maximum(weights, 1e-8), 0, 1)


def edge_map(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb.astype(np.uint8)
    return (cv2.Canny(gray, 60, 140) > 0)


def depth_discontinuity(depth, valid_mask=None, percentile=88.0):
    d = np.asarray(depth, np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    valid = np.isfinite(d) if valid_mask is None else np.asarray(valid_mask, bool) & np.isfinite(d)
    threshold = float(np.percentile(mag[valid], percentile)) if valid.any() else np.inf
    return (mag >= max(threshold, 1e-6)) & valid, mag


def knn_fill(rgb, sparse, valid_mask, k=5, spatial_sigma=4.0, color_sigma=0.12):
    """Local RGB-aware KNN interpolation over valid sparse samples.

    Coordinates are evaluated in the input grid. RGB is resized to that grid so
    neighbour colour similarity can reduce cross-boundary depth mixing.
    """
    d = np.asarray(sparse, np.float32); m = np.asarray(valid_mask, bool) & np.isfinite(d)
    h, w = d.shape
    if not m.any(): raise ValueError("KNN requires at least one valid depth sample")
    if m.all(): return d.copy()
    out = d.copy()
    yy, xx = np.indices((h, w)); valid_xy = np.column_stack([yy[m], xx[m]]).astype(np.float32)
    valid_values = d[m]
    guide = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    if guide.max(initial=0) > 1: guide /= 255.0
    valid_colors = guide[m]
    tree = cKDTree(valid_xy)
    missing = ~m
    query_xy = np.column_stack([yy[missing], xx[missing]]).astype(np.float32)
    dist, idx = tree.query(query_xy, k=min(k, len(valid_values)), workers=1)
    if dist.ndim == 1: dist, idx = dist[:, None], idx[:, None]
    query_colors = guide[missing]
    color_dist = np.sum((query_colors[:, None, :] - valid_colors[idx])**2, axis=2)
    weights = np.exp(-(dist**2)/(2*spatial_sigma**2+1e-8) - color_dist/(2*color_sigma**2+1e-8))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    out[missing] = np.sum(weights * valid_values[idx], axis=1).astype(np.float32)
    return np.clip(out, 0, 1)


def classical_refine(rgb, depth, method="guided", valid_mask=None):
    d = np.asarray(depth, np.float32)
    valid = np.isfinite(d) & (d > 0) if valid_mask is None else (np.asarray(valid_mask, bool) & np.isfinite(d))
    if not valid.any(): return np.zeros_like(d)
    filled = fill_holes(d, valid)
    # Median denoising is deliberately mild; measured values are restored below.
    smooth_input = cv2.medianBlur(np.clip(filled, 0, 1).astype(np.float32), 3)
    if method == "guided": refined = guided_filter(rgb, smooth_input)
    elif method == "joint_bilateral": refined = joint_bilateral(rgb, smooth_input)
    else: raise ValueError(f"Unknown classical filter: {method}")
    boundary, grad = depth_discontinuity(filled, np.ones_like(valid, dtype=bool))
    rgb_edges = edge_map(rgb)
    threshold = float(np.percentile(grad[np.isfinite(grad)], 88)) if np.isfinite(grad).any() else 1.0
    # RGB edges contribute only where there is also geometric depth evidence.
    rgb_depth_support = rgb_edges & (grad >= max(threshold * 0.5, 1e-6))
    strength = np.maximum(boundary.astype(np.float32), rgb_depth_support.astype(np.float32))
    smoothed = cv2.GaussianBlur(refined, (0, 0), 0.8)
    out = strength * refined + (1-strength) * smoothed
    # Preserve known samples exactly.
    out[valid] = d[valid]
    return np.clip(out, 0, 1)
