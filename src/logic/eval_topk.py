import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]


def build_model(num_classes: int):
    model = torchvision.models.densenet121(weights=None)
    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, num_classes)
    return model


def get_eval_transform(size: int = 224):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def load_rows(test_csv: Path) -> List[Dict[str, str]]:
    with test_csv.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description="Study-level top-k aggregation evaluator")
    ap.add_argument("--processed-root", required=True, help="Processed data root containing test/<class>/*.jpg")
    ap.add_argument("--test-csv", required=True, help="splits/test.csv from mapping")
    ap.add_argument("--model", required=True, help="Path to trained weights (best_model.pt)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--topk", type=int, default=5, help="Number of top slices to aggregate per study")
    args = ap.parse_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    transform = get_eval_transform(args.size)
    model = build_model(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    processed_root = Path(args.processed_root)
    rows = load_rows(Path(args.test_csv))

    # Predict slice probabilities
    study_to_probs: Dict[str, List[np.ndarray]] = defaultdict(list)
    study_to_label: Dict[str, int] = {}

    for r in rows:
        cls = r["class_label"]
        fname = Path(r["image_path"]).name
        img_path = processed_root / "test" / cls / fname
        if not img_path.exists():
            # skip missing (should not happen)
            continue
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x).cpu().numpy()
        probs = softmax_np(logits)[0]
        sid = r["pseudo_study_id"]
        study_to_probs[sid].append(probs)
        study_to_label[sid] = CLASSES.index(cls)

    # Aggregate with top-k selection by slice confidence (max prob)
    y_true: List[int] = []
    y_pred: List[int] = []
    for sid, prob_list in study_to_probs.items():
        arr = np.stack(prob_list, axis=0)  # [N, C]
        conf = arr.max(axis=1)
        k = min(args.topk, arr.shape[0])
        top_idx = np.argsort(-conf)[:k]
        agg = arr[top_idx].mean(axis=0)
        pred = int(agg.argmax())
        y_pred.append(pred)
        y_true.append(study_to_label[sid])

    print(f"Evaluated {len(y_true)} studies using top-{args.topk} aggregation.")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()


