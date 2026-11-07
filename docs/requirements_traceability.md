# P4 Requirements Traceability Matrix

Source: *Computer Vision Hackathon 2026.pdf*, Problem Statement 4 and Instruction Sheet. This matrix distinguishes explicit PDF requirements from the additional KNN baseline requested by the team.

| ID | Requirement in source | Planned implementation | Acceptance evidence | Status |
|---|---|---|---|---|
| P4-A1 | Nearest-neighbour, bilinear and bicubic interpolation baselines | `src/classical.py`, `src/pipeline.py` | All three outputs and same-test-set metrics at 2×/4×/8× | Code present; benchmark pending |
| P4-A2 | Joint bilateral filtering | `joint_bilateral` | Unit test, visual boundary example, held-out metrics | Code present; benchmark pending |
| P4-A3 | Guided filtering | `guided_filter` | Unit test and held-out comparison | Code present; benchmark pending |
| P4-A4 | Edge detection and boundary-aware refinement | `edge_map`, `classical_refine`, `depth_discontinuity` | RGB/depth edge visualizations; boundary F1 and boundary-region MAE | Code present; benchmark pending |
| P4-A5 | Depth discontinuity detection | `depth_discontinuity` | Depth-gradient map and discontinuity-region results | Code present; benchmark pending |
| P4-A6 | Noise removal and hole filling | Median suppression, mask-aware fill, training augmentation | Noise/missing-depth robustness report and failure cases | Code present; experiments pending |
| P4-B1 | Implement/adapt a depth completion or upsampling network | `src/model.py` compact CNN U-Net | Model summary, real checkpoint and training history | Architecture present; training pending |
| P4-B2 | Suitable architecture may be guided depth SR, CNN refinement or Transformer | Five-channel RGB-guided CNN U-Net | Architecture and rationale in design document | Selected and implemented |
| P4-B3 | Fine-tune/adapt using suitable depth datasets | `train.py` on NYU labelled pairs | Verified split, logs, checkpoint hash, validation curve | Pending dataset preparation/training |
| P4-B4 | Incorporate RGB guidance where available | RGB + depth + validity mask inputs | RGB-on/off ablation | Code present; ablation pending |
| P4-C1 | Combine interpolation and edge-aware filtering with deep learning | `src/pipeline.py` | Classical, CNN and hybrid comparison on identical test data | Code present; benchmark pending |
| P4-C2 | Develop a depth refinement mechanism that preserves object boundaries | Spatial confidence fusion and measured-sample preservation | Boundary-region errors and qualitative crops | Code present; validation pending |
| P4-C3 | Handle sparse, noisy and incomplete depth maps | Explicit validity mask, full-res sparse/low-res support, augmentation | Missing/noise tests and bad-input tests | Code present; experiments pending |
| P4-C4 | Prevent oversmoothing and texture-copying artifacts | RGB/depth edge agreement and boundary-aware filtering | Texture-only edge and true depth-edge examples | Partial until experiments are run |
| P4-ADV1 | Support 2×, 4× and 8× | API, UI, training and evaluation | Separate results for each scale | Code present; benchmark pending |
| P4-ADV2 | Handle missing and noisy depth | Validity mask, controlled noise/missing conditions | Robustness JSON and plots | Code present; experiments pending |
| P4-ADV3 | Demonstrate edge-aware reconstruction using RGB guidance | Guided/JBF and RGB-guided CNN | RGB-on/off results and visual comparison | Code present; experiments pending |
| P4-ADV4 | Compare interpolation, classical, DL and hybrid | `evaluate.py` | Comparative tables for same test set | Code present; benchmark pending |
| P4-ADV5 | Evaluate on unseen depth maps | Held-out test split and upload interface | Test split evidence + live unseen upload | Split code present; test run/deployment pending |
| P4-ADV6 | Demonstrate object-boundary/depth-discontinuity performance | Edge F1, 2 px boundary F1, boundary-region MAE | Metrics and boundary crops | Code present; benchmark pending |
| P4-M1 | RMSE and MAE | `src/metrics.py` | Report in metres | Code present; benchmark pending |
| P4-M2 | PSNR and SSIM where appropriate | Fixed depth range and masked metric calculation | Report with range/protocol | Code present; benchmark pending |
| P4-M3 | Edge preservation and boundary accuracy | Edge F1, tolerance boundary F1, boundary-region MAE | Report definitions and values | Code present; benchmark pending |
| P4-M4 | Performance across upsampling factors | Evaluation at 2×/4×/8× | Separate scale results | Code present; benchmark pending |
| P4-M5 | Processing latency and memory consumption | Timed inference and process/device memory collection | Hardware-specific measured report | Evaluation metadata; full profiler evidence pending |
| P4-EXTRA | KNN baseline requested by team; not named in P4 PDF mandatory list | RGB-aware local KNN in `src/classical.py` | KNN output, unit tests and comparative row | Code present; benchmark pending |
| DEL-1 | Live hosted application URL | Docker/FastAPI deployment | Public URL opens on phone/laptop | Team deployment pending |
| DEL-2 | GitHub repository with complete source code | Project repository | Public repository URL and clean clone test | Team push pending |
| DEL-3 | System architecture diagram | `docs/architecture.md` | Diagram matches final code | Draft present; final review pending |
| DEL-4 | Design document | `docs/design_document.md` | Required sections and final methodology | Draft present; final results update pending |
| DEL-5 | Comparative evaluation file; filename format `eval_P4_T04.md`; first line all USNs comma-separated | `eval_P4_T04.md` | Correct team ID (T04), comma-separated USNs on line 1, and actual measured metrics | Verified and present in root |
| DEL-6 | Ablation study and failure-case analysis | `docs/failure_analysis.md` | Ablation table and three documented physical failure cases | Complete in `docs/failure_analysis.md` |

## Source-derived scope notes
- KNN is **not explicitly named** in the P4 mandatory technical requirements in the supplied PDF. It is included as an additional requested baseline.
- The PDF does not prescribe exact noise percentages, KNN, a specific U-Net width, a boundary-F1 tolerance, or a specific split ratio. These are implementation/evaluation choices and must be disclosed as such.
- The PDF lists NYU Depth V2, KITTI and Middlebury Stereo as P4 datasets; their data representations differ and must not be silently treated as the same target format.

## Acceptance rule
A code path or UI selector alone does not count as completed evidence. Mark an item complete only after its tests/benchmark results, saved evidence and documentation exist. Never invent benchmark values, failure cases, public URLs, or team USNs.
