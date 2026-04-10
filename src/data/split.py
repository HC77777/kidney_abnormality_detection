#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import random


Row = Tuple[str, str, str, int]


def read_mapping(csv_path: Path) -> List[Row]:
    rows: List[Row] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                (
                    r["image_path"],
                    r["class_label"],
                    r["pseudo_study_id"],
                    int(r["index_within_study"]),
                )
            )
    if not rows:
        raise SystemExit(f"No rows in mapping: {csv_path}")
    return rows


def split_groups(
    rows: List[Row], train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> Dict[str, List[str]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Build groups per class
    class_to_groups: Dict[str, List[str]] = {}
    for _, cls, sid, _ in rows:
        class_to_groups.setdefault(cls, set()).add(sid)
    # Convert sets to sorted lists
    for cls in list(class_to_groups.keys()):
        class_to_groups[cls] = sorted(class_to_groups[cls])

    rng = random.Random(seed)

    split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for cls, groups in class_to_groups.items():
        groups = groups.copy()
        rng.shuffle(groups)
        n = len(groups)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # ensure sum equals n
        n_train = min(n_train, n)
        n_val = min(n_val, max(0, n - n_train))
        n_test = n - n_train - n_val
        split["train"].extend(groups[:n_train])
        split["val"].extend(groups[n_train : n_train + n_val])
        split["test"].extend(groups[n_train + n_val :])
    return split


def write_split_csv(
    rows: List[Row], split: Dict[str, List[str]], out_dir: Path
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    sid_to_split: Dict[str, str] = {}
    for s in split["train"]:
        sid_to_split[s] = "train"
    for s in split["val"]:
        sid_to_split[s] = "val"
    for s in split["test"]:
        sid_to_split[s] = "test"

    paths = {
        "train": out_dir / "train.csv",
        "val": out_dir / "val.csv",
        "test": out_dir / "test.csv",
    }
    writers: Dict[str, csv.writer] = {}
    files = {}
    try:
        for k, p in paths.items():
            f = p.open("w", newline="")
            files[k] = f
            w = csv.writer(f)
            writers[k] = w
            w.writerow(["image_path", "class_label", "pseudo_study_id", "index_within_study"])  # header

        for r in rows:
            img, cls, sid, idx = r
            s = sid_to_split[sid]
            writers[s].writerow([img, cls, sid, idx])
    finally:
        for f in files.values():
            f.close()

    return paths["train"], paths["val"], paths["test"]


def summarize(paths: Tuple[Path, Path, Path]) -> None:
    for name, p in zip(["train", "val", "test"], paths):
        with p.open("r", newline="") as f:
            reader = csv.DictReader(f)
            n = 0
            by_class: Dict[str, int] = {}
            for r in reader:
                n += 1
                by_class[r["class_label"]] = by_class.get(r["class_label"], 0) + 1
        print(f"{name}: {n} images")
        for cls, cnt in sorted(by_class.items()):
            print(f"  {cls}: {cnt}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Split dataset by pseudo_study_id")
    ap.add_argument("--mapping-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_mapping(Path(args.mapping_csv))
    split = split_groups(rows, args.train, args.val, args.test, args.seed)
    paths = write_split_csv(rows, split, Path(args.output_dir))
    summarize(paths)


if __name__ == "__main__":
    main()


