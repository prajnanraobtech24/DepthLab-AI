# Official Rubric Readiness — P4

The criterion maxima in the PDF's rubric table sum to 100 points:

| Criterion | Maximum | Evidence to protect marks |
|---|---:|---|
| Evaluation Metrics | 25 | Reproducible held-out metrics, same inputs across methods, 2×/4×/8×, boundary results, robustness, ablations, latency and memory |
| Implementation | 20 | Core methods on unseen inputs, proper train/validation/test separation, modular code, bad-input handling, real trained checkpoint |
| UI/UX and Deployment | 15 | Public URL works first click on phone and laptop, about 10-second response target, loading/error states, one-command run or Docker, no secrets |
| Project Documentation | 15 | Architecture matches code, complete design document, reproducible README, limitations and ethics stated honestly |
| Demo and Viva | 25 | Clear three-minute live demo on URL with judge's unseen input; every member explains own part, model/classical choices, metrics and failure case |

The instruction sheet has a prose sentence about “each of the other five criteria” carrying 15 points, but the actual table has five rows with maxima 25, 20, 15, 15 and 25, which total 100. Follow the actual table maxima.

## Explicit deductions/caps
- Late submission: minus 5 points for every 15 minutes beyond deadline.
- No live URL: UI/UX and deployment capped at 0.
- No GitHub link: Implementation capped at 0.
- Missing architecture diagram or design document: Project documentation capped at 5.
- Tie-break: higher Demo and Viva score, then higher Implementation score.

## Submission evidence checklist
- [ ] Public hosted application URL tested from a phone and laptop.
- [ ] GitHub repository contains complete source code.
- [ ] Architecture diagram reflects the deployed code.
- [ ] Design document covers problem, design, methodology, datasets, evaluation, limitations and ethics.
- [ ] `eval_P4_TeamID.md` has all USNs comma-separated on its first line and actual metrics in P4 metric order.
- [ ] Comparative results: interpolation, classical, CNN and hybrid; include the team's additional KNN baseline.
- [ ] Ablation study with actual values.
- [ ] At least three observed failure cases, with input/output/GT/error evidence and explanation.
- [ ] README commands reproduce preparation, training and evaluation.
- [ ] No raw dataset, credentials or fabricated results in GitHub.

## Three-minute demo
- 0:00–0:25 — Explain sparse/low-resolution depth and aligned RGB guidance.
- 0:25–1:05 — Upload unseen input, choose 4×, run a selected pipeline.
- 1:05–1:40 — Compare interpolation, KNN, classical, CNN and hybrid outputs; show a boundary crop.
- 1:40–2:15 — Demonstrate a missing/noisy-depth case or an additional scale.
- 2:15–2:45 — Show one real failure case and explain its cause and mitigation.
- 2:45–3:00 — State measured results, limitations and team contributions.
