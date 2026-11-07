"""
Utility to turn ANY custom photo (phone camera, downloaded picture, screenshot)
into a paired RGB + Depth set ready for manual drag-and-drop upload into DepthLab AI.

Usage:
    python create_custom_depth_pair.py path/to/your_image.jpg
    
Output (saved in 'my_custom_uploads/'):
    1. <name>_rgb.png          -> Drop into 'RGB guidance image'
    2. <name>_depth_4x.png     -> Drop into 'Sparse / Low-Res depth map' (Scale: 4x)
    3. <name>_depth_sparse.png -> Drop into 'Sparse / Low-Res depth map' (Input Type: Full-resolution sparse)
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def process_image(image_path: str, output_dir: str = "my_custom_uploads"):
    path = Path(image_path)
    if not path.is_file():
        print(f"Error: File not found: {image_path}")
        return

    out_folder = Path(output_dir)
    out_folder.mkdir(parents=True, exist_ok=True)

    # 1. Read input image
    img = cv2.imread(str(path))
    if img is None:
        print(f"Error: Could not decode image: {image_path}")
        return

    # Convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Ensure dimensions are divisible by 8 and comfortably sized (e.g., max 640x480)
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        w = int(w * scale)
        h = int(h * scale)
    w = (w // 8) * 8
    h = (h // 8) * 8
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

    stem = path.stem

    # 2. Save standardized RGB image
    rgb_save_path = out_folder / f"{stem}_rgb.png"
    Image.fromarray(rgb).save(rgb_save_path, format="PNG")
    print(f"[1/3] Saved RGB Image: {rgb_save_path} ({w}x{h})")

    # 3. Synthesize realistic depth map from geometric priors (ground plane + luminance + edges)
    # Background further away (4.5m), foreground objects closer (1.2m - 2.5m)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # Vertical perspective gradient (top = distant ceiling/sky, bottom = close floor)
    y_coords = np.linspace(0.8, 0.2, h)[:, None]
    base_depth = 1.0 + 3.5 * y_coords  # 1.0m to 4.5m
    
    # Modulate by structure & inverted luminance (darker/recessed areas further, objects closer)
    depth_m = base_depth + (1.0 - gray) * 0.8
    depth_m = cv2.bilateralFilter(depth_m.astype(np.float32), d=9, sigmaColor=0.5, sigmaSpace=15.0)
    depth_m = np.clip(depth_m, 0.5, 6.0)

    # 4. Generate 4x Low-Resolution Depth Map (h//4, w//4)
    target_lh, target_lw = h // 4, w // 4
    low_res_depth = cv2.resize(depth_m, (target_lw, target_lh), interpolation=cv2.INTER_AREA)
    # Add a small simulated specular void (e.g. glass or sensor blind spot)
    void_y, void_x = target_lh // 3, target_lw // 3
    low_res_depth[void_y:void_y + 4, void_x:void_x + 5] = 0.0

    # Save as 16-bit millimeter PNG (standard depth format)
    depth_uint16 = np.clip(np.rint(low_res_depth * 1000.0), 0, 65535).astype(np.uint16)
    depth_save_path = out_folder / f"{stem}_depth_4x.png"
    Image.fromarray(depth_uint16).save(depth_save_path, format="PNG")
    print(f"[2/3] Saved 4x Low-Res Depth: {depth_save_path} ({target_lw}x{target_lh})")

    # 5. Generate Sparse LiDAR Simulation (5% random laser returns at full resolution)
    sparse_depth = np.zeros((h, w), dtype=np.float32)
    # Random uniform sampling (5% density)
    np.random.seed(42)
    sample_mask = np.random.rand(h, w) < 0.05
    sparse_depth[sample_mask] = depth_m[sample_mask]
    
    sparse_uint16 = np.clip(np.rint(sparse_depth * 1000.0), 0, 65535).astype(np.uint16)
    sparse_save_path = out_folder / f"{stem}_depth_sparse.png"
    Image.fromarray(sparse_uint16).save(sparse_save_path, format="PNG")
    print(f"[3/3] Saved Sparse LiDAR Depth: {sparse_save_path} ({w}x{h}, 5% density)")

    print("\n" + "="*60)
    print("READY TO UPLOAD!")
    print(f"1. In DepthLab AI, click 'RGB guidance image' -> Select: {rgb_save_path}")
    print(f"2. Click 'Sparse / Low-Res depth map'      -> Select: {depth_save_path}")
    print(f"3. Under 'DEPTH FILE UNIT', select: Millimetres (mm)")
    print(f"4. Under 'UPSAMPLING FACTOR', select: 4x (Standard)")
    print(f"5. Click 'Run reconstruction'!")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_custom_depth_pair.py <path_to_image.jpg>")
        print("Example: python create_custom_depth_pair.py test.jpg")
    else:
        process_image(sys.argv[1])
