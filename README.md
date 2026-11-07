# DepthLab AI — Edge-Preserving Depth Map Upsampling & Refinement

**Hackathon problem:** ProbID P4 — Edge-Preserving Depth Map Upsampling and Refinement.

DepthLab AI reconstructs dense high-resolution depth from a low-resolution or full-resolution sparse depth map, optionally guided by an aligned RGB image. It provides interpolation baselines, RGB-aware KNN, joint bilateral and guided filtering, a compact CNN U-Net, and confidence-adaptive hybrid fusion.

## System Architecture

![DepthLab AI System Architecture](assets/system_architecture.png)

*Interactive FigJam Architecture Board:* [DepthLab AI - System Architecture on FigJam](https://www.figma.com/board/RWWzVX9Fie6uvLxEGQ2yyU)

> **Important:** This source package does not contain NYU/KITTI/Middlebury datasets or trained weights. It does not claim benchmark results before you run the prepared datasets and training/evaluation scripts. The application works in classical mode before a checkpoint is trained.

## 1. Dataset placement — do this once

After extracting this project ZIP, place your downloaded files exactly here. Do not put the datasets in `src/`, `web/`, or the project root.

```text
depthlab_ai/
└── data/
    ├── downloads/
    │   ├── nyu/
    │   │   └── nyu_depth_v2_labeled.mat       # your NYU MAT file (keep as MAT)
    │   ├── kitti/
    │   │   ├── data_depth_velodyne.zip
    │   │   ├── data_depth_annotated.zip
    │   │   ├── data_depth_selection.zip
    │   │   └── devkit_depth.zip
    │   └── middlebury/
    │       └── All.zip
    ├── raw/
    │   ├── nyu/
    │   ├── kitti/
    │   └── middlebury/
    └── processed/
        ├── nyu/
        ├── kitti/
        └── middlebury/
```

Create these folders if they do not exist. Keep the original NYU `.mat` file and all downloaded ZIP archives in `data/downloads/`. Extract the four KITTI ZIP files into the **same** `data/raw/kitti/` directory, preserving their internal folder names. Extract `All.zip` into `data/raw/middlebury/`. Do not extract the NYU MAT file. `data/raw/nyu/` can remain empty; the converter reads the MAT directly. Keep `data/processed/` empty until the preparation scripts create output.

The archive deliberately excludes all datasets and trained checkpoints. `.gitignore` keeps dataset contents, checkpoints and experiment results out of GitHub; the `data/README.md` and empty folder markers remain.

## 2. Environment and run the website

Python 3.10 or 3.11 is recommended. On Windows PowerShell, from the `depthlab_ai` folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. API docs: `http://127.0.0.1:8000/docs`.

Alternative: run `run_windows.bat` on Windows or `bash run_local.sh` on Linux/macOS.

## 3. Inspect the datasets before conversion

First inventory the downloads and extracted folders:

```bash
python inspect_datasets.py --root data/downloads
python inspect_datasets.py --root data/raw
```

### NYU Depth V2 — primary training dataset

The MAT file is **not extracted**. First inspect the array names and shapes:

```bash
python prepare_nyu.py --mat data/downloads/nyu/nyu_depth_v2_labeled.mat --inspect-only
```

Then, only after the displayed `images`/`depths` dimensions look plausible and the arrays are correctly identified:

```bash
python prepare_nyu.py --mat data/downloads/nyu/nyu_depth_v2_labeled.mat --out data/processed/nyu
```

The converter supports HDF5/MAT v7.3 and classic MAT when readable by SciPy. It validates sample counts and expected 480×640 RGB/depth orientation and refuses unknown shapes. It creates `train/val/test/{rgb,gt}` NPY pairs and `split_metadata.json`. The current default is a deterministic image-level 70/15/15 split; it is **not guaranteed scene-disjoint**. Do not claim scene-level separation. Inspect a few saved pairs and confirm RGB/depth alignment and that depth is in metres before training.

### KITTI Depth Completion — outdoor sparse-LiDAR evaluation

Extract each of the four archives into the same `data/raw/kitti/` folder. Preserve each archive's internal paths. Then inventory and create a candidate manifest:

```bash
python inspect_datasets.py --root data/raw
python prepare_kitti.py --root data/raw/kitti --out data/processed/kitti_manifest.csv
```

Open the CSV and verify a few RGB/sparse/GT rows refer to the same frame and split. Only after checking the paths, convert complete triplets:

```bash
python prepare_kitti.py --root data/raw/kitti --out data/processed/kitti_manifest.csv --convert
python evaluate_kitti.py --data data/processed/kitti --split val
```

The manifest is a candidate index, not proof of correct pairing. KITTI projected depth PNG values are conventionally stored at **256 units per metre**; the converter decodes `metres = uint16_value / 256.0`, with zero meaning invalid. Do not treat zero as physical depth or raw values as millimetres. Keep sparse Velodyne input and annotated ground truth separate. The KITTI evaluation script reports sparse/KNN/guided/joint-bilateral classical results in metres. It deliberately does not apply the NYU-trained CNN with an incompatible 0.1–10 m normalization to 0.1–80 m KITTI data. Respect official validation/test restrictions; do not use hidden test ground truth.

### Middlebury Stereo — disparity-based additional evaluation

Extract `All.zip` into `data/raw/middlebury/`, then run:

```bash
python prepare_middlebury.py --root data/raw/middlebury --out data/processed/middlebury_manifest.csv
```

The script inventories PFM files and prints their dimensions. Middlebury is disparity-centric, not a directly metric RGB-D training source. Do **not** treat disparity as metres. Metric conversion requires the scene's calibration and the correct disparity convention: `Z = (f × B) / d`. Invalid/occluded disparity values must be excluded. If calibration cannot be verified, report disparity-space geometric results separately instead of inventing metric depth.

## 4. Train the CNN U-Net

The selected network is a compact five-channel RGB-guided U-Net: RGB (3) + filled upsampled depth (1) + observation-validity mask (1). It is a CNN, not a Transformer. The network is trained on synthetic 2×/4×/8× inputs generated from dense NYU targets. Downsampling is validity-weighted so invalid zero values do not contaminate the target. Training augmentation introduces missing samples and mild metric noise.

```bash
python train.py --data data/processed/nyu --epochs 30 --batch-size 4
```

The best validation checkpoint is written to `checkpoints/depthrefinenet.pt`; training history is written to `checkpoints/training_history.json`. If memory is limited, reduce batch size to 1 or 2. The script is CPU-compatible but training can be slow. Record actual runtime, hardware, seed, epoch count and validation scores. Never describe random/untrained weights as a trained model.

## 5. Evaluate and run robustness experiments

```bash
python evaluate.py --data data/processed/nyu --checkpoint checkpoints/depthrefinenet.pt
python evaluate_robustness.py --data data/processed/nyu --checkpoint checkpoints/depthrefinenet.pt --limit 50
```

The main evaluation compares nearest, bilinear, bicubic, KNN, guided filter, joint bilateral filter, CNN U-Net and hybrid reconstruction at 2×/4×/8× on the same held-out test images. `--limit 5` is a smoke test only; do not present it as the complete benchmark. The robustness script tests clean input, 10/30/50% missing measurements and 0.02/0.05 m additive noise. The random seed and sample count are saved in the report.

Metrics:
- RMSE and MAE in **metres** over valid ground-truth pixels.
- PSNR with a fixed depth range, not per-image normalization.
- SSIM with fixed normalization and valid-region treatment.
- Edge F1 and 2-pixel tolerance boundary F1.
- Boundary-region MAE and non-boundary MAE (boundary band derived from GT depth gradients).
- Median/mean inference latency and available memory measurements.

For a fair comparison, use identical test samples, RGB/depth alignment, fixed depth range, GT validity mask and hardware. Do not tune using the test split. The current NYU split is image-level and must be disclosed. KITTI and Middlebury results are separate domain-specific evaluations; do not average incomparable datasets into one score.

## 6. Website input behavior

- RGB must be an aligned PNG/JPEG guidance image.
- Depth may be a lower-resolution depth map or a full-resolution sparse map.
- `Auto-detect` treats equal RGB/depth dimensions as full-resolution sparse; otherwise it treats depth as low-resolution. You can select the input type explicitly.
- Depth unit must be selected correctly. NYU profile is 0.1–10 m; KITTI profile is 0.1–80 m. Convert KITTI PNG encodings to metres before upload; raw KITTI PNG values are not millimetres.
- Optional GT must match RGB dimensions exactly and use the selected units.
- Modes: interpolation, KNN, classical, deep learning and hybrid. If no trained checkpoint exists, DL/hybrid safely fall back to classical output and the UI warns you.
- Output PNG is uint16 with zero reserved for invalid: millimetres for ranges up to 65.535 m, or KITTI 1/256-m encoding for the 80 m profile. The API reports the encoding. NPY output always stores metric metres.

## 7. Project structure

```text
server.py                    FastAPI app and input validation
src/classical.py             NN/bilinear/bicubic helpers, KNN, guided/JBF, edges, holes
src/model.py                 Compact RGB-guided CNN U-Net
src/pipeline.py              Full-resolution sparse/low-resolution pipeline + adaptive fusion
src/metrics.py                Metric-depth, SSIM/PSNR, edge and boundary metrics
prepare_nyu.py                MAT inspection and NYU NPY conversion
prepare_kitti.py              KITTI pairing manifest and optional NPY conversion
evaluate_kitti.py             KITTI sparse/classical cross-domain evaluation
prepare_middlebury.py         PFM reader and Middlebury inventory
inspect_datasets.py           Read-only dataset inventory
train.py                      Validity-aware U-Net training
evaluate.py                   Held-out comparative evaluation
evaluate_robustness.py        Missing/noise robustness tests
web/                          Responsive browser interface
docs/                          Architecture, design, protocol, traceability and rubric docs
tests/                         Unit and API tests
```

## 8. Test the project

```bash
python -m pytest -q
```

Tests cover interpolation, masks, KNN, guided/joint-bilateral filtering, hole filling, physical metrics, API inputs and full-resolution sparse-depth handling. Tests do not prove that a model has trained or that benchmark performance is competitive; those require the actual dataset runs.

## 9. Deployment

```bash
docker build -t depthlab-ai .
docker run --rm -p 8000:8000 depthlab-ai
```

Deploy the Docker image or Python app to a public host, then verify the URL from a phone and laptop. For a local Docker build after training, the checkpoint in `checkpoints/` is included in the build context (but remains Git-ignored). For a host that builds from GitHub, upload the checkpoint through the host's private artifact/persistent storage and set `DEPTHLAB_CHECKPOINT` to its mounted path. A public URL is not generated by this source archive. Do not commit API keys, secrets, datasets or trained checkpoints to GitHub.

## 10. Mandatory submission checklist

- [ ] Live hosted application URL tested from phone and laptop.
- [ ] Public GitHub repository with complete source code.
- [x] Architecture diagram matches the final code (`assets/system_architecture.png` & `docs/architecture.md`).
- [x] Design document has methodology, datasets, implementation, evaluation, limitations and ethics (`docs/design_document.md`).
- [x] `eval_P4_T04.md`: first line contains all team USNs (`1RUA24CSE0243,1RUA24CSE0263,1RUA24CSE0270,1RUA24CSE0308`), comma-separated, followed by comprehensive comparative metrics across 2×, 4×, and 8× scales.
- [ ] Comparative results for interpolation, classical, CNN and hybrid, including KNN baseline.
- [ ] 2×/4×/8× results, MAE/RMSE, PSNR/SSIM, edge/boundary accuracy, latency and memory.
- [ ] Ablation results: RGB off, mask off, classical-only, CNN-only and hybrid.
- [ ] At least three **observed** failure cases with evidence, causes and mitigations.
- [ ] README reproduces setup and evaluation; limitations and ethics stated honestly.
- [ ] Three-minute live demo rehearsed; every member can explain their component, model, classical choices, metrics and a failure case.

See `docs/requirements_traceability.md` and `docs/rubric_readiness.md` for the requirement-by-requirement acceptance evidence.
