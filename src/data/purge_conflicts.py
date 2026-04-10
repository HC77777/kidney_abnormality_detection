import hashlib
from pathlib import Path
from collections import defaultdict
import sys
import os

def get_file_hash(path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def purge_conflicts(root_dir: Path):
    print(f"Scanning {root_dir} for conflicts to PURGE...")
    
    content_map = defaultdict(list)
    files = list(root_dir.rglob("*.*"))
    
    for p in files:
        if p.is_file() and p.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            h = get_file_hash(p)
            # Extract class (parent folder)
            cls = p.parent.name
            content_map[h].append((cls, p))
            
    deleted_count = 0
    conflict_groups = 0
    
    for h, entries in content_map.items():
        classes = set(e[0] for e in entries)
        
        if len(classes) > 1:
            conflict_groups += 1
            print(f"Conflict [{h[:8]}]: Found in {classes}. DELETING ALL copies.")
            for cls, p in entries:
                try:
                    os.remove(p)
                    print(f"  Deleted: {p}")
                    deleted_count += 1
                except OSError as e:
                    print(f"  Error deleting {p}: {e}")

    print("\n" + "="*40)
    print(f"Purge Complete.")
    print(f"Conflict Groups Found: {conflict_groups}")
    print(f"Total Files Deleted: {deleted_count}")
    print("="*40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python purge_conflicts.py <data_root>")
        sys.exit(1)
    
    purge_conflicts(Path(sys.argv[1]))

