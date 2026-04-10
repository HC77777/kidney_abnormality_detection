import os
from pathlib import Path
from typing import Callable, List, Tuple, Optional

from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, classes: List[str], transform: Optional[Callable] = None):
        self.root = Path(root_dir)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

        self.samples: List[Tuple[Path, int]] = []
        for c in classes:
            for p in sorted((self.root / c).glob("*.jpg")):
                self.samples.append((p, self.class_to_idx[c]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


