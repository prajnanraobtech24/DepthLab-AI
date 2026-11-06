# Dataset Protocol — P4

## 1. Official dataset list in the hackathon brief
The P4 datasheet lists **NYU Depth V2**, **KITTI dataset**, and **Middlebury Stereo**. These are not identical data formats and are assigned distinct roles below.

## 2. Local directory layout

```text
data/
├── downloads/
│   ├── nyu/          # original .mat file, not extracted
│   ├── kitti/        # four original ZIP files
│   └── middlebury/   # All.zip
├── raw/
│   ├── nyu/          # may remain empty; MAT is read directly
│   ├── kitti/        # extract all four KITTI ZIPs into this one directory
│   └── middlebury/   # extract All.zip here
└── processed/
    ├── nyu/          # train/val/test RGB/GT NPY arrays
    ├── kitti/        # candidate pairing manifest and future prepared arrays
    └── middlebury/   # scene/PFM manifest
```

Keep downloads intact. Do not extract NYU MAT. Extract KITTI archives into the same raw KITTI directory while preserving their internal paths. Extract Middlebury All.zip into raw Middlebury. Raw and processed dataset files are local-only and ignored by Git.

## 3. NYU Depth V2 — primary indoor training/evaluation
- Official source: https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
- Use the labelled MAT file containing paired RGB images and dense depth maps.
- First run `python prepare_nyu.py --mat data/downloads/nyu/nyu_depth_v2_labeled.mat --inspect-only`.
- Confirm `images` and `depths` keys, sample counts, array shapes and expected 480×640 orientation. The converter supports HDF5/MAT v7.3 and classic MAT when SciPy can read it; ambiguous shapes are rejected.
- Convert using `python prepare_nyu.py --mat data/downloads/nyu/nyu_depth_v2_labeled.mat --out data/processed/nyu`.
- Output: `train/val/test/{rgb,gt}/*.npy`, with RGB uint8 H×W×3 and depth float32 H×W.
- Default split: deterministic image-level 70/15/15, seed 42. This is **not scene-disjoint** unless a verified scene identifier is added. Do not claim scene-level generalization.
- Confirm depth units from the exact source file before training. NYU labelled depth is expected in metres; do not scale blindly.
- Primary training uses dense depth as GT and creates validity-weighted synthetic 2×/4×/8× low-resolution inputs. Test data remains held out.

## 4. KITTI Depth Completion — outdoor sparse-LiDAR evaluation
- Official source: https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
- Expected downloaded archives from the user's download set: `data_depth_velodyne.zip`, `data_depth_annotated.zip`, `data_depth_selection.zip`, and `devkit_depth.zip`.
- Place originals in `data/downloads/kitti/`; extract all into `data/raw/kitti/` without renaming internal folders.
- Run `python prepare_kitti.py --root data/raw/kitti --out data/processed/kitti_manifest.csv`.
- Review candidate pairs manually. The manifest script uses relative path keys and prints duplicate warnings, but archive layouts can differ; it is not proof of correct pairing.
- After verifying complete RGB/sparse/GT triplets, run `python prepare_kitti.py --root data/raw/kitti --out data/processed/kitti_manifest.csv --convert`. This creates split folders containing `rgb/`, `sparse/` and `gt/` NPY arrays.
- Run `python evaluate_kitti.py --data data/processed/kitti --split val` for sparse/KNN/guided/joint-bilateral cross-domain evaluation. The NYU-trained CNN is intentionally not evaluated with a mismatched depth normalization range; any KITTI CNN adaptation must be trained and evaluated with a separate 0.1–80 m protocol.
- Projected KITTI depth PNG values are conventionally stored at 256 units per metre: `depth_m = uint16_value / 256.0`. A zero value means no measurement. Do not treat raw pixel values as millimetres.
- Keep Velodyne sparse input separate from annotated ground truth. Respect official train/validation/test rules and never use hidden test ground truth.
- Initially use KITTI as a separate outdoor robustness/cross-domain evaluation, not mixed into NYU training.

## 5. Middlebury Stereo — disparity-based geometric evaluation
- Official sources: https://vision.middlebury.edu/stereo/data/ and https://vision.middlebury.edu/stereo/submit3/
- Place `All.zip` in `data/downloads/middlebury/`; extract to `data/raw/middlebury/`.
- Run `python prepare_middlebury.py --root data/raw/middlebury --out data/processed/middlebury_manifest.csv`.
- The utility reads PFM headers, dimensions, endianness and scale, and lists scene files. It does not convert disparity to depth automatically.
- Metric conversion requires scene-specific calibration and disparity convention: `Z = (f × B) / d`. Exclude invalid/occluded disparities. If calibration cannot be verified, report disparity-space results separately, not metres.
- Initially use Middlebury for additional geometric/boundary evaluation rather than primary CNN training.

## 6. Training input generation and validity rules
- Fixed NYU metric range: 0.1–10.0 m; document any justified changes.
- Valid GT: finite, positive and within the selected range. Preserve the validity mask separately from normalized values.
- For resizing/downsampling, resize `depth × mask` and `mask` separately, divide numerator by denominator, and apply a documented valid-support threshold. Never average invalid zeros into depth.
- For the network, use RGB (3) + filled upsampled depth (1) + observation mask (1). The filled depth channel is not a substitute for the mask.
- Test set must not be used for training, hyperparameter selection, checkpoint selection or threshold selection.

## 7. Required experiment protocol
- Evaluate each method on identical held-out samples at 2×, 4× and 8×.
- Methods: nearest, bilinear, bicubic, additional RGB-aware KNN, guided filter, joint bilateral, CNN U-Net and hybrid.
- Robustness conditions: clean; 10%, 30%, 50% missing input samples; additive noise with standard deviations 0.02 m and 0.05 m. These are project-defined settings, not values prescribed by the PDF.
- Ablations: RGB guidance off, validity mask off, interpolation/classical/CNN/hybrid comparison.
- At least three observed failure cases must be captured from real runs.

## 8. Metrics and units
- RMSE and MAE in metres over valid GT and valid prediction pixels; report prediction coverage so missing outputs are visible.
- PSNR uses a fixed 0.1–10.0 m depth range for NYU; SSIM uses the same fixed normalization and valid-window policy.
- Edge F1 and boundary F1 (2-pixel tolerance) use depth-gradient edges. Boundary-region MAE uses a 5×5 dilated GT boundary band.
- Latency: one warm-up, then median of repeated inference runs. Record image dimensions and hardware.
- Memory: report process RSS, peak CPU RSS where available, peak GPU allocation, model parameter count and checkpoint size.
- Do not average raw NYU, KITTI and Middlebury metrics unless the representations, units, masks and protocols have been made comparable and the averaging is justified.

## 9. Reproducibility and licensing
Record exact release, download source, file checksum, split manifest, depth units, scale, missing/noise seed, model checkpoint SHA-256, code commit, hardware/software and commands. Do not commit raw datasets or redistribute processed data unless the dataset terms allow it.
