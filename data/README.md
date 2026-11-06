# Dataset placement

Place original downloads under `downloads/`:
- `downloads/nyu/`: original NYU labelled `.mat` file (do not extract).
- `downloads/kitti/`: four original KITTI ZIP archives.
- `downloads/middlebury/`: original `All.zip` archive.

Extract KITTI archives into the same `raw/kitti/` directory. Extract Middlebury `All.zip` into `raw/middlebury/`. NYU MAT is read directly; `raw/nyu/` may remain empty. Preparation scripts write generated arrays/manifests into `processed/`. Keep all dataset contents local; do not commit them.
