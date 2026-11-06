# Design Document — DepthLab AI

## 1. Problem statement and objective
P4 asks for an intelligent edge-aware depth reconstruction system that converts sparse or low-resolution depth into dense high-resolution depth while preserving object boundaries, geometric structure and depth discontinuities. It must address missing values, sensor noise, foreground/background discontinuities, texture artifacts and inaccurate boundaries. RGB guidance should improve reconstruction without copying unrelated RGB texture into depth.

## 2. Scope and users
- A judge uploads an unseen RGB/depth pair and inspects the reconstructed map.
- A researcher compares interpolation, classical CV, deep learning and hybrid methods.
- A team member evaluates a dense reference, studies boundary errors and exports metric depth.

The application does not claim to recover ground truth for arbitrary scenes. It estimates depth from the provided inputs and learned training distribution.

## 3. Functional requirements
- Upload aligned RGB and either low-resolution or full-resolution sparse depth.
- Accept optional dense ground truth for quantitative evaluation.
- Support 2×, 4× and 8× scale settings.
- Implement nearest-neighbour, bilinear and bicubic interpolation baselines.
- Implement RGB-aware KNN as an additional comparison baseline.
- Implement joint bilateral filtering and guided filtering separately.
- Detect RGB edges and depth discontinuities; use boundary-aware refinement.
- Handle missing depth, noise and holes with an explicit validity mask.
- Implement/train a depth completion/upsampling network with RGB guidance.
- Combine interpolation, edge-aware refinement and CNN output in a hybrid method.
- Compare outputs and report depth metrics, boundary accuracy, latency and memory.
- Export uint16 PNG with a declared encoding (millimetres for ranges up to 65.535 m; KITTI 1/256 m for the 80 m profile) and NPY in metres.

## 4. Non-functional requirements
- FastAPI backend; HTML/CSS/JavaScript responsive frontend.
- Modular source, deterministic data split seed and documented dataset protocol.
- Invalid input rejected with useful messages; no silent resizing of ground truth.
- No random/untrained checkpoint presented as learned inference.
- No raw datasets, credentials or generated benchmark values committed to GitHub.

## 5. Methodology
### 5.1 Input representation and preprocessing
Depth values are converted to metres and clipped to a documented dataset range. Invalid samples are tracked in a boolean mask, never inferred from a normalized value of zero. Full-resolution sparse depth and genuinely low-resolution depth are handled as separate input types. Mask-aware interpolation resizes the weighted depth numerator and validity denominator separately to prevent invalid zeros from contaminating valid measurements.

### 5.2 Classical computer vision
- **Interpolation:** nearest-neighbour, bilinear and bicubic.
- **KNN (additional baseline):** local spatial neighbours with RGB colour similarity; known samples are preserved.
- **Joint bilateral:** RGB colour and spatial distances control the weights applied to depth.
- **Guided filter:** local linear model uses RGB-derived guidance.
- **Edges and discontinuities:** RGB Canny edges and depth-gradient edges are computed separately. RGB edges strengthen a boundary only when supported by geometric depth gradients.
- **Noise and holes:** mild median suppression and nearest-valid hole filling; large holes remain a documented failure risk.

### 5.3 Deep learning
A compact CNN U-Net is selected instead of a Transformer. The P4 brief permits CNN-based refinement or transformer-based models; it does not mandate a Transformer. The U-Net is a practical training/deployment choice for a short hackathon: encoder/decoder feature extraction with skip connections, modest memory use and a clear training protocol.

Input channels: RGB (3), filled interpolated depth (1), validity mask (1). Output: fixed-range normalized dense depth. Training uses dense NYU targets and synthetic 2×/4×/8× low-resolution inputs. Validity-weighted downsampling prevents invalid target zeros from contaminating coarse depth. Augmentation introduces missing coarse measurements and additive metric noise. Loss is masked L1 plus a depth-gradient term. The best checkpoint is selected using validation performance. Test data is never used for model selection.

### 5.4 Hybrid reconstruction
A spatial confidence map is derived from local depth gradients and boundary evidence. A configurable base learned-branch contribution is reduced near strong depth boundaries, where the classical edge-aware result receives more weight. Reliable full-resolution measured values are restored after reconstruction. This is a transparent heuristic and must be ablated; it is not described as a learned uncertainty estimator.

## 6. Datasets and split protocol
- **NYU Depth V2:** primary indoor RGB-D training and held-out evaluation. Inspect MAT keys/shapes, verify RGB/depth alignment and units. Current default split is deterministic image-level 70/15/15; it is not scene-disjoint unless scene identifiers are incorporated.
- **KITTI Depth Completion:** outdoor sparse LiDAR cross-domain evaluation. Keep Velodyne input and annotated ground truth distinct; decode PNG values as depth metres = pixel/256; zero is invalid. Validate pair identity and official split.
- **Middlebury Stereo:** additional stereo/disparity evaluation. Parse PFM and calibration. Do not convert disparity to metric depth without correct focal length, baseline and disparity convention.

The three datasets are reported separately. Metrics from incompatible depth/disparity protocols are not pooled.

## 7. Evaluation protocol
- Same held-out RGB/depth/GT samples, valid mask, depth range and scale for every compared method.
- RMSE and MAE in metres; PSNR with fixed metric range; SSIM with fixed normalization.
- Edge F1 and 2-pixel tolerance boundary F1; boundary-region MAE uses a 5×5 dilated GT depth-edge band.
- Results at 2×/4×/8×; latency after warm-up; CPU and GPU memory separately where available.
- Robustness: clean, 10/30/50% missing and 0.02/0.05 m noise.
- Ablations: RGB off, mask off, interpolation, classical-only, CNN-only and hybrid.
- At least three actual failure cases documented with evidence, cause and mitigation.

## 8. UI/UX and deployment
The interface uses a dark technical visual theme, drag-and-drop inputs, scale and method controls, explicit depth unit/range, loading and error feedback, interpolation, KNN, classical, CNN, hybrid and boundary views plus an optional GT error map, metric cards and depth exports. It is responsive for mobile and desktop. Docker and local startup commands are included. A public URL must be deployed and tested separately; it is not included in the source archive.

## 9. Limitations and ethics
- Depth estimates can be unreliable on reflective/transparent surfaces, thin objects, large holes, motion, occlusion and RGB-depth misalignment.
- RGB texture does not always indicate geometry; texture-copying must be tested.
- NYU indoor training may not generalize to outdoor KITTI without adaptation.
- Middlebury disparity cannot be compared directly with metric depth.
- KNN uses a capped support subset on dense maps and may trade exact neighbour search for speed.
- The compact U-Net is a project model, not a clinically/safety validated depth sensor replacement.
- Do not publish dataset files if the dataset licence prohibits redistribution.

## 10. Reproducibility
Record dataset release/source, file checksums, split manifest, units/range, random seeds, training history, checkpoint SHA-256, code commit, hardware/software, input resolution, scale, timing protocol, metric masks and exact commands. README provides reproducible setup, preprocessing, training and evaluation commands.
