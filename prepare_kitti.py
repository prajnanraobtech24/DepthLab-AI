
"""Inventory and convert paired KITTI Depth Completion samples.

Matches RGB, sparse depth and ground-truth files using normalized
sequence/frame filenames. KITTI uint16 depth is decoded as value / 256 metres.
"""
from pathlib import Path
import argparse
import csv
import json
import re

import numpy as np
from PIL import Image


MODALITIES = {
    "image",
    "image_2",
    "image_02",
    "velodyne_raw",
    "groundtruth",
    "ground_truth",
    "groundtruth_depth",
    "proj_depth",
}


def classify(p, root):
    parts = [x.lower() for x in p.relative_to(root).parts]
    filename = p.name.lower()

    # Check ground truth and sparse depth before RGB because filenames
    # may also contain the word "image" near the end.
    if (
        any(x in {"groundtruth", "ground_truth", "groundtruth_depth"} for x in parts)
        or "groundtruth_depth" in filename
        or any(x.startswith("data_depth_annotated") for x in parts)
    ):
        return "ground_truth"

    if (
        "velodyne_raw" in parts
        or "velodyne_raw" in filename
        or any(x.startswith("data_depth_velodyne") for x in parts)
    ):
        return "sparse"

    if (
        any(x in {"image", "image_2", "image_02"} for x in parts)
        or any(x.startswith("data_rgb") for x in parts)
    ):
        return "rgb"

    return "other"


def key(p, root):
    parts = list(p.relative_to(root).parts)

    # Remove modality-specific directories.
    parts = [
        x for x in parts
        if x.lower() not in MODALITIES
        and not x.lower().startswith(("data_depth_", "devkit_depth"))
    ]

    # Normalize modality-specific filename tokens.
    # Example:
    # ..._image_0000000005_image_02.png
    # ..._velodyne_raw_0000000005_image_02.png
    # ..._groundtruth_depth_0000000005_image_02.png
    #
    # All three become the same frame key.
    filename = parts[-1]
    path = Path(filename)

    stem = re.sub(
        r"_(?:image|velodyne_raw|groundtruth_depth)_(?=\d{10}(?:_|$))",
        "_",
        path.stem,
        flags=re.IGNORECASE,
    )

    parts[-1] = stem + path.suffix.lower()

    return "/".join(parts)


def split_for_key(k):
    parts = [x.lower() for x in k.split("/")]

    if any("val" in x or "validation" in x for x in parts):
        return "val"

    if "train" in parts:
        return "train"

    if "test" in parts:
        return "test"

    return "unassigned"


def read_kitti_depth(path):
    raw = np.asarray(Image.open(path))

    if raw.ndim == 3:
        raw = raw[..., 0]

    if raw.dtype != np.uint16:
        raise ValueError(
            f"Expected uint16 KITTI depth PNG: {path} ({raw.dtype})"
        )

    depth = raw.astype(np.float32) / 256.0
    depth[raw == 0] = 0.0

    return depth


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", default="data/raw/kitti")
    ap.add_argument("--out", default="data/processed/kitti_manifest.csv")
    ap.add_argument(
        "--convert",
        action="store_true",
        help="Convert complete RGB/sparse/GT triplets to NPY",
    )
    ap.add_argument("--data-out", default="data/processed/kitti")

    args = ap.parse_args()

    root = Path(args.root)

    if not root.exists():
        raise SystemExit(
            f"KITTI raw directory not found: {root.resolve()}"
        )

    groups = {
        "rgb": {},
        "sparse": {},
        "ground_truth": {},
    }

    # Inventory files.
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue

        category = classify(p, root)

        if category not in groups:
            continue

        sample_key = key(p, root)

        if sample_key in groups[category]:
            print(
                f"WARNING: duplicate {category} key: {sample_key}"
            )
            continue

        groups[category][sample_key] = p

    # Create a combined manifest.
    all_keys = sorted(
        set(groups["rgb"])
        | set(groups["sparse"])
        | set(groups["ground_truth"])
    )

    rows = []

    for sample_key in all_keys:
        rgb = groups["rgb"].get(sample_key)
        sparse = groups["sparse"].get(sample_key)
        gt = groups["ground_truth"].get(sample_key)

        rows.append({
            "key": sample_key,
            "split": split_for_key(sample_key),
            "rgb": rgb,
            "sparse": sparse,
            "gt": gt,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "key",
            "split",
            "rgb_path",
            "sparse_depth_path",
            "ground_truth_path",
        ])

        for r in rows:
            writer.writerow([
                r["key"],
                r["split"],
                str(r["rgb"] or ""),
                str(r["sparse"] or ""),
                str(r["gt"] or ""),
            ])

    complete = [
        r for r in rows
        if r["rgb"]
        and r["sparse"]
        and r["gt"]
        and r["split"] in ("train", "val", "test")
    ]

    val_complete = [
        r for r in complete if r["split"] == "val"
    ]

    print("\n========== KITTI INVENTORY ==========")
    print(f"RGB files: {len(groups['rgb'])}")
    print(f"Sparse files: {len(groups['sparse'])}")
    print(f"Ground-truth files: {len(groups['ground_truth'])}")
    print(f"Candidate keys: {len(rows)}")
    print(f"Complete triplets: {len(complete)}")
    print(f"Complete validation triplets: {len(val_complete)}")
    print(f"Manifest: {out.resolve()}")

    if not args.convert:
        print("\nInventory only. Review the manifest before conversion.")
        return

    # Convert only complete, recognizable triplets.
    data_out = Path(args.data_out)
    converted = 0
    skipped = 0

    for i, r in enumerate(complete):
        try:
            rgb = np.asarray(
                Image.open(r["rgb"]).convert("RGB")
            )

            sparse = read_kitti_depth(r["sparse"])
            gt = read_kitti_depth(r["gt"])

            if (
                rgb.shape[:2] != sparse.shape
                or rgb.shape[:2] != gt.shape
            ):
                raise ValueError(
                    f"Dimension mismatch: "
                    f"RGB={rgb.shape}, sparse={sparse.shape}, GT={gt.shape}"
                )

            sid = f"kitti_{i:06d}"
            base = data_out / r["split"]

            for sub in ("rgb", "sparse", "gt"):
                (base / sub).mkdir(parents=True, exist_ok=True)

            np.save(base / "rgb" / f"{sid}.npy", rgb)
            np.save(base / "sparse" / f"{sid}.npy", sparse)
            np.save(base / "gt" / f"{sid}.npy", gt)

            converted += 1

        except Exception as exc:
            print(f"SKIP {r['key']}: {exc}")
            skipped += 1

    metadata = {
        "source_root": str(root.resolve()),
        "converted_triplets": converted,
        "skipped_triplets": skipped,
        "depth_units": "metres",
        "png_encoding": "uint16 value / 256 = metres; zero invalid",
        "split_note": (
            "Splits inferred from source directory names. "
            "Validation uses val_selection_cropped where applicable."
        ),
    }

    data_out.mkdir(parents=True, exist_ok=True)

    (data_out / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n========== CONVERSION COMPLETE ==========")
    print(f"Converted triplets: {converted}")
    print(f"Skipped triplets: {skipped}")
    print(f"Output directory: {data_out.resolve()}")


if __name__ == "__main__":
    main()
