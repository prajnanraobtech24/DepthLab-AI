# Three-minute demo and viva preparation

## Live demo sequence (3 minutes)
1. **Problem (0:00–0:25):** “P4 asks us to recover a dense high-resolution depth map from sparse or low-resolution depth, while retaining object boundaries and avoiding RGB texture-copying.”
2. **Input (0:25–0:45):** Upload an aligned RGB image and a matching depth map. State whether it is low-resolution or full-resolution sparse, select units and choose 4×.
3. **Methods (0:45–1:20):** Run the app and explain interpolation, RGB-aware KNN, guided/joint-bilateral classical refinement, the trained CNN U-Net and adaptive hybrid. Explicitly state if the checkpoint is trained; never present fallback as DL.
4. **Evidence (1:20–1:55):** Show the comparison cards, depth-boundary map, ground-truth error map and actual metrics. Explain RMSE/MAE in metres and the 2-pixel boundary tolerance.
5. **Robustness (1:55–2:25):** Show a held-out missing/noisy sample or a different scale. State the exact test condition and sample count.
6. **Failure (2:25–2:50):** Show one actual failure case and explain its cause, metric impact and mitigation.
7. **Close (2:50–3:00):** State one measured result, one limitation and each member's contribution.

## Viva questions to rehearse

**Why CNN U-Net and not a Transformer?**  
The P4 brief allows CNN-based refinement or transformer-based depth models; it does not mandate a Transformer. The compact U-Net has encoder/decoder features and skip connections, is practical to train and deploy within the hackathon time, and matches the five-channel RGB/depth/mask input. We should justify it using measured results, not claim it is universally superior.

**What does the validity mask do?**  
It distinguishes a missing sensor measurement from a valid depth that becomes zero after fixed-range normalization. It is passed as a separate input channel and used to exclude invalid GT pixels from loss and metrics.

**How is KNN different from nearest-neighbour interpolation?**  
Nearest-neighbour interpolation copies one nearest grid sample based on image coordinates. Our KNN baseline queries multiple valid depth samples and combines them with weights based on spatial distance and RGB colour similarity. KNN is an additional project baseline; it is not explicitly named in the P4 mandatory list in the supplied PDF.

**What is joint bilateral filtering?**  
It filters the depth signal while computing weights from spatial proximity and RGB guidance similarity. It can preserve supported edges better than ordinary smoothing, but can copy texture if RGB and depth boundaries disagree.

**What is guided filtering?**  
It models the output locally as a linear transform of the guidance image. It uses RGB structure to refine depth while controlling smoothing through a local regularization parameter.

**How do you prevent texture copying?**  
RGB edges alone are not considered depth boundaries. We require supporting depth-gradient evidence and evaluate boundary accuracy and texture-only-edge failure cases.

**How does the hybrid method work?**  
It combines the classical edge-aware estimate and CNN estimate. The learned contribution is reduced around strong depth boundaries; measured full-resolution samples are restored after fusion. The fusion is a heuristic and is evaluated through ablation.

**How are NYU, KITTI and Middlebury different?**  
NYU supplies indoor RGB-D dense labels for primary training. KITTI supplies outdoor RGB and sparse LiDAR depth with separate annotated GT and its own encoding/splits. Middlebury is stereo disparity and requires calibration for metric depth. Their metrics are reported separately.

**What does RMSE measure?**  
The square root of mean squared metric-depth error over valid evaluation pixels, in metres. It penalizes large errors more than MAE.

**What does PSNR/SSIM mean here?**  
They compare reconstructed and reference depth using a fixed depth range and consistent valid-region handling. They are supplementary to physical MAE/RMSE and boundary metrics.

**What is the main limitation?**  
The NYU-trained model may not generalize to outdoor KITTI. Large holes, thin structures, reflective surfaces and RGB-depth misalignment can also cause errors. We disclose measured failures rather than hiding them.

**How do you prove unseen-input support?**  
Use the live URL with an RGB/depth pair not used for training or tuning, record the exact sample provenance, and show the output and validation behavior.
