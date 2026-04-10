import argparse
import csv
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

# Import the advanced pipeline logic
from preprocess_advanced import preprocess_pipeline

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def run_dataset(csv_path: Path, out_root: Path, subset_name: str, project_root: Path, size: int):
    """
    Reads the split CSV and processes every image using the Advanced Pipeline.
    """
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Processing {subset_name} ({len(rows)} images)...")
    
    success_count = 0
    
    for r in tqdm(rows):
        # Resolve source path
        rel_path = Path(r["image_path"])
        if rel_path.is_absolute():
             # Try to make it relative if it matches project root
             try:
                 rel_path = rel_path.relative_to(project_root)
             except ValueError:
                 pass # It's some other absolute path, keep it
                 
        src_path = project_root / rel_path
        cls = r["class_label"]
        
        # Define destination path: data/ct-kidney/processed/{subset}/{class}/{filename}
        dst_path = out_root / subset_name / cls / rel_path.name
        ensure_dir(dst_path.parent)
        
        try:
            # Run the Advanced Pipeline
            # (Smart Crop -> CLAHE -> Isotropic Resize -> RGB)
            processed_img = preprocess_pipeline(str(src_path), target_size=size)
            
            # Save
            cv2.imwrite(str(dst_path), processed_img)
            success_count += 1
        except Exception as e:
            print(f"Failed to process {src_path}: {e}")

    print(f"Finished {subset_name}: {success_count}/{len(rows)} images successfully processed.")

def main():
    ap = argparse.ArgumentParser(description="Run Advanced Preprocessing on the entire dataset")
    ap.add_argument("--splits-dir", required=True, help="Folder containing train.csv, val.csv, test.csv")
    ap.add_argument("--output-dir", required=True, help="Where to save processed images")
    ap.add_argument("--project-root", default=str(Path.cwd()))
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.output_dir)
    project_root = Path(args.project_root)

    # Process each split
    for split in ["train", "val", "test"]:
        csv_file = splits_dir / f"{split}.csv"
        if not csv_file.exists():
            print(f"Skipping {split}, CSV not found at {csv_file}")
            continue
            
        run_dataset(csv_file, out_dir, split, project_root, args.size)

if __name__ == "__main__":
    main()

