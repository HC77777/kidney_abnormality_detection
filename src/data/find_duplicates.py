import hashlib
from pathlib import Path
from collections import defaultdict
import sys

def get_file_hash(path: Path) -> str:
    """Returns MD5 hash of file content."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def find_duplicates(root_dir: Path):
    print(f"Scanning {root_dir} for duplicates...")
    
    # hash -> list of (class, path)
    content_map = defaultdict(list)
    
    # Walk through all files
    files = list(root_dir.rglob("*.*"))
    print(f"Found {len(files)} files. Hashing...")
    
    for p in files:
        if p.is_file() and p.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            h = get_file_hash(p)
            # Extract class name (parent folder)
            # processed/train/Stone/img.jpg -> class=Stone
            cls = p.parent.name
            content_map[h].append((cls, p))
            
    # Check for collisions
    conflict_count = 0
    duplicates_count = 0
    
    for h, entries in content_map.items():
        if len(entries) > 1:
            duplicates_count += 1
            classes = set(e[0] for e in entries)
            if len(classes) > 1:
                conflict_count += 1
                print(f"\n🚨 CONFLICT FOUND (Hash: {h[:8]}):")
                for cls, p in entries:
                    print(f"  - [{cls}] {p}")

    print("\n" + "="*40)
    print(f"Total Files: {len(files)}")
    print(f"Total Duplicate Groups: {duplicates_count}")
    print(f"CRITICAL Conflicts (Same Image, Different Class): {conflict_count}")
    print("="*40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_duplicates.py <data_root>")
        sys.exit(1)
    
    find_duplicates(Path(sys.argv[1]))

