#!/usr/bin/env python3

import argparse
import csv
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def apply_clahe(image_gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_gray)


def heuristic_kidney_roi(image_gray: np.ndarray) -> Tuple[int, int, int, int]:
    # Normalize to 0-255
    img = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX)
    # Median blur to reduce noise
    blurred = cv2.medianBlur(img, 5)
    # Otsu threshold
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Keep largest connected component as rough body region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if num_labels <= 1:
        h, w = image_gray.shape
        return 0, 0, w, h
    # Largest component (skip label 0 which is background)
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h = stats[largest_idx, :4]
    # Expand slightly
    pad_x = int(0.05 * w)
    pad_y = int(0.05 * h)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(image_gray.shape[1], x + w + pad_x)
    y1 = min(image_gray.shape[0], y + h + pad_y)
    return x0, y0, x1 - x0, y1 - y0


def process_image(
    src_path: Path,
    dst_path: Path,
    output_size: Tuple[int, int],
    use_roi: bool,
    do_clahe: bool,
) -> bool:
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    if do_clahe:
        img = apply_clahe(img)
    if use_roi:
        x, y, w, h = heuristic_kidney_roi(img)
        img = img[y : y + h, x : x + w]
    img = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)
    # Normalize to 0-1 then back to 0-255 for saving, preserving contrast stretch
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    ensure_dir(dst_path.parent)
    return cv2.imwrite(str(dst_path), img)


def run(csv_in: Path, out_root: Path, subset_name: str, size: int, roi: bool, clahe: bool, project_root: Path) -> None:
    with csv_in.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    total = len(rows)
    ok = 0
    for r in rows:
        rel_img = Path(r["image_path"]) if not Path(r["image_path"]).is_absolute() else Path(r["image_path"]).relative_to(project_root)
        src = project_root / rel_img
        cls = r["class_label"]
        dst = out_root / subset_name / cls / rel_img.name
        if process_image(src, dst, (size, size), roi, clahe):
            ok += 1
    print(f"{subset_name}: processed {ok}/{total} images -> {out_root / subset_name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess images: resize, CLAHE, optional ROI")
    ap.add_argument("--splits-dir", required=True, help="Directory with train.csv/val.csv/test.csv")
    ap.add_argument("--output-dir", required=True, help="Output root for processed images")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--roi", action="store_true", help="Apply heuristic ROI crop")
    ap.add_argument("--clahe", action="store_true", help="Apply CLAHE normalization")
    ap.add_argument("--project-root", default=str(Path.cwd()))
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()

    for subset in ["train", "val", "test"]:
        csv_in = splits_dir / f"{subset}.csv"
        if not csv_in.exists():
            raise SystemExit(f"Missing split file: {csv_in}")
        run(csv_in, out_root, subset, args.size, args.roi, args.clahe, project_root)


if __name__ == "__main__":
    main()


