#!/usr/bin/env python3

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple


def find_class_folders(root_dir: Path) -> Dict[str, Path]:
    class_dirs: Dict[str, Path] = {}
    for entry in sorted(root_dir.iterdir()):
        if entry.is_dir():
            class_dirs[entry.name] = entry
    if not class_dirs:
        raise SystemExit(f"No class folders found under: {root_dir}")
    return class_dirs


def collect_images(class_dir: Path, extensions: Tuple[str, ...]) -> List[Path]:
    images: List[Path] = []
    for ext in extensions:
        images.extend(sorted(class_dir.rglob(f"*{ext}")))
    return images


def chunk_list(items: List[Path], chunk_size: int) -> List[List[Path]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def build_mapping(
    root_dir: Path,
    group_size: int,
    seed: int,
    extensions: Tuple[str, ...],
) -> List[Tuple[Path, str, str, int]]:
    rng = random.Random(seed)
    class_dirs = find_class_folders(root_dir)

    mapping: List[Tuple[Path, str, str, int]] = []
    for class_name, class_path in class_dirs.items():
        images = collect_images(class_path, extensions)
        if not images:
            print(f"Warning: no images for class '{class_name}' in {class_path}")
            continue
        rng.shuffle(images)
        groups = chunk_list(images, group_size)
        for group_index, group in enumerate(groups):
            pseudo_study_id = f"{class_name}_ps{group_index:05d}"
            for index_within_study, img_path in enumerate(group):
                mapping.append((img_path, class_name, pseudo_study_id, index_within_study))
    return mapping


def write_csv(mapping: List[Tuple[Path, str, str, int]], output_csv: Path, base_path: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "class_label", "pseudo_study_id", "index_within_study"])
        for img_path, class_label, pseudo_id, idx in mapping:
            try:
                rel = img_path.relative_to(base_path)
            except ValueError:
                rel = img_path
            writer.writerow([str(rel), class_label, pseudo_id, idx])


def summarize(mapping: List[Tuple[Path, str, str, int]]) -> None:
    by_class: Dict[str, int] = {}
    by_study: Dict[str, int] = {}
    for _, cls, sid, _ in mapping:
        by_class[cls] = by_class.get(cls, 0) + 1
        by_study[sid] = by_study.get(sid, 0) + 1
    print("Total images:", len(mapping))
    print("Images by class:")
    for cls, n in sorted(by_class.items()):
        print(f"  {cls}: {n}")
    print("Number of pseudo studies:", len(by_study))
    lengths = sorted(by_study.values())
    if lengths:
        print(
            "Study size (min/median/max):",
            lengths[0],
            lengths[len(lengths) // 2],
            lengths[-1],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pseudo study IDs for slice images")
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Path containing class subfolders with images",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=20,
        help="Number of images per pseudo study (last group may be smaller)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--ext",
        nargs="*",
        default=[".jpg", ".jpeg", ".png"],
        help="Image extensions to include",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Where to write the mapping CSV",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path.cwd()),
        help="Base path to make image paths relative in the CSV",
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    extensions = tuple(x.lower() if x.startswith(".") else f".{x.lower()}" for x in args.ext)

    if not root_dir.exists():
        raise SystemExit(f"Root dir does not exist: {root_dir}")

    mapping = build_mapping(root_dir, args.group_size, args.seed, extensions)
    write_csv(mapping, output_csv, project_root)
    summarize(mapping)
    print(f"\nWrote mapping to: {output_csv}")


if __name__ == "__main__":
    main()


