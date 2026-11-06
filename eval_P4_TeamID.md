USN_1,USN_2,USN_3,USN_4

# P4 Comparative Evaluation — replace TeamID and the first-line USN placeholders

**Dataset:** NYU Depth V2 labelled set (record exact release and split manifest)  
**Depth range:** 0.1–10.0 m  
**Split:** deterministic image-level 70/15/15 unless a verified scene-disjoint split is prepared; do not claim scene-disjoint separation otherwise.  
**Hardware/software:** [CPU, GPU, RAM, OS, Python, PyTorch, OpenCV]  
**Checkpoint:** [path, SHA-256, trained epoch, validation result]  
**Test sample count:** [N]  
**Input generation:** validity-weighted area downsampling; report missing/noise condition and random seed.

> No values may be filled with estimates or demo screenshots. Run `evaluate.py` and copy measured values from `results/evaluation.json`. Report each dataset separately.

## 1. RMSE and MAE (metres)

| Scale | Method | Filter | RMSE (m) ↓ | MAE (m) ↓ | N |
|---|---|---|---:|---:|---:|
| 2× | Nearest | — | — | — | — |
| 2× | Bilinear | — | — | — | — |
| 2× | Bicubic | — | — | — | — |
| 2× | KNN | RGB-aware local KNN | — | — | — |
| 2× | Classical | Guided | — | — | — |
| 2× | Classical | Joint bilateral | — | — | — |
| 2× | CNN U-Net | — | — | — | — |
| 2× | Hybrid | Adaptive fusion | — | — | — |
| 4× | Nearest | — | — | — | — |
| 4× | Bilinear | — | — | — | — |
| 4× | Bicubic | — | — | — | — |
| 4× | KNN | RGB-aware local KNN | — | — | — |
| 4× | Classical | Guided | — | — | — |
| 4× | Classical | Joint bilateral | — | — | — |
| 4× | CNN U-Net | — | — | — | — |
| 4× | Hybrid | Adaptive fusion | — | — | — |
| 8× | Nearest | — | — | — | — |
| 8× | Bilinear | — | — | — | — |
| 8× | Bicubic | — | — | — | — |
| 8× | KNN | RGB-aware local KNN | — | — | — |
| 8× | Classical | Guided | — | — | — |
| 8× | Classical | Joint bilateral | — | — | — |
| 8× | CNN U-Net | — | — | — | — |
| 8× | Hybrid | Adaptive fusion | — | — | — |

## 2. PSNR and SSIM

Use fixed 0.1–10.0 m range for every method and image.

| Scale | Method | PSNR (dB) ↑ | SSIM ↑ |
|---|---|---:|---:|
| 2× | Nearest | — | — |
| 2× | Bilinear | — | — |
| 2× | Bicubic | — | — |
| 2× | KNN | — | — |
| 2× | Guided filter | — | — |
| 2× | Joint bilateral | — | — |
| 2× | CNN U-Net | — | — |
| 2× | Hybrid | — | — |
| 4× | Nearest | — | — |
| 4× | Bilinear | — | — |
| 4× | Bicubic | — | — |
| 4× | KNN | — | — |
| 4× | Guided filter | — | — |
| 4× | Joint bilateral | — | — |
| 4× | CNN U-Net | — | — |
| 4× | Hybrid | — | — |
| 8× | Nearest | — | — |
| 8× | Bilinear | — | — |
| 8× | Bicubic | — | — |
| 8× | KNN | — | — |
| 8× | Guided filter | — | — |
| 8× | Joint bilateral | — | — |
| 8× | CNN U-Net | — | — |
| 8× | Hybrid | — | — |

## 3. Edge preservation and boundary accuracy

State edge detector, thresholding, valid mask, tolerance and boundary-band construction. Current implementation uses depth-gradient edges, 2-pixel tolerance for boundary F1, and a 5×5 dilation of GT depth edges for boundary-region MAE.

| Scale | Method | Edge F1 ↑ | Boundary F1 (2 px) ↑ | Boundary MAE (m) ↓ | Non-boundary MAE (m) ↓ |
|---|---|---:|---:|---:|---:|
| 2× | Nearest | — | — | — | — |
| 2× | Bilinear | — | — | — | — |
| 2× | Bicubic | — | — | — | — |
| 2× | KNN | — | — | — | — |
| 2× | Guided filter | — | — | — | — |
| 2× | Joint bilateral | — | — | — | — |
| 2× | CNN U-Net | — | — | — | — |
| 2× | Hybrid | — | — | — | — |
| 4× | Nearest | — | — | — | — |
| 4× | Bilinear | — | — | — | — |
| 4× | Bicubic | — | — | — | — |
| 4× | KNN | — | — | — | — |
| 4× | Guided filter | — | — | — | — |
| 4× | Joint bilateral | — | — | — | — |
| 4× | CNN U-Net | — | — | — | — |
| 4× | Hybrid | — | — | — | — |
| 8× | Nearest | — | — | — | — |
| 8× | Bilinear | — | — | — | — |
| 8× | Bicubic | — | — | — | — |
| 8× | KNN | — | — | — | — |
| 8× | Guided filter | — | — | — | — |
| 8× | Joint bilateral | — | — | — | — |
| 8× | CNN U-Net | — | — | — | — |
| 8× | Hybrid | — | — | — | — |

## 4. Performance across upsampling factors

Discuss how errors and boundary scores change from 2× to 4× to 8×. Include the actual output examples and identify the best method per scale without cherry-picking.

## 5. Processing latency and memory consumption

| Method | Scale | Median latency (ms) | Peak CPU RAM (MB) | Peak GPU memory (MB) | Device |
|---|---:|---:|---:|---:|---|
| [measured method] | 2×/4×/8× | — | — | — | — |

Record warm-up count, repeat count, image dimensions, batch size, precision and timing implementation. Distinguish per-request latency from training time.

## 6. Ablation study

| Variant | RGB guidance | Validity mask | Classical branch | CNN branch | RMSE (m) | MAE (m) | Boundary F1 |
|---|---|---|---|---|---:|---:|---:|
| Interpolation | No | No | No | No | — | — | — |
| Classical | Yes | Yes | Yes | No | — | — | — |
| CNN full | Yes | Yes | No | Yes | — | — | — |
| CNN RGB off | No | Yes | No | Yes | — | — | — |
| CNN mask off | Yes | No | No | Yes | — | — | — |
| Hybrid | Yes | Yes | Yes | Yes | — | — | — |

## 7. Robustness and failure-case analysis

Report clean, 10%, 30%, 50% missing-depth and 0.02 m / 0.05 m noise results. Include at least three actual failure cases with input, output, ground truth, error map, cause and mitigation. Do not fill this section with hypothetical failures.

## 8. KITTI and Middlebury additional evaluation

Report each separately with its own split, depth/disparity convention, valid-pixel policy and units. KITTI projected depth PNG uses 256 units per metre. Middlebury disparity must not be reported as metres unless calibration conversion is verified. Do not merge these results with NYU averages.

## 9. Reproducibility

- Git commit: [hash]
- Checkpoint SHA-256: [hash]
- Dataset release/source and local manifest checksum: [details]
- Commands: [exact commands]
- Random seeds: [values]
- Limitations and ethics: [honest statement]
