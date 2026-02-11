import argparse
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from data.datasets import ImageFolderDataset
from model.efficientnet import build_efficientnet_v2


def get_transforms(size: int = 224):
    train_tf = T.Compose([
        T.RandomResizedCrop(size, scale=(0.9, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),  # Added stronger augmentation
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def run_train(args):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    classes = ["Cyst", "Normal", "Stone", "Tumor"]

    train_tf, eval_tf = get_transforms(args.size)

    train_ds = ImageFolderDataset(str(Path(args.data_root)/"train"), classes, transform=train_tf)
    val_ds = ImageFolderDataset(str(Path(args.data_root)/"val"), classes, transform=eval_tf)
    test_ds = ImageFolderDataset(str(Path(args.data_root)/"test"), classes, transform=eval_tf)

    # Optional quick subset for sanity runs
    if args.limit_samples > 0:
        random.seed(42)
        def subset_indices(n_total: int, n_limit: int):
            n = min(n_total, n_limit)
            return random.sample(range(n_total), n)
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, subset_indices(len(train_ds), args.limit_samples))
        val_ds = Subset(val_ds, subset_indices(len(val_ds), max(1, args.limit_samples // 4)))
        test_ds = Subset(test_ds, subset_indices(len(test_ds), max(1, args.limit_samples // 4)))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Building EfficientNetV2-S model for {len(classes)} classes...")
    model = build_efficientnet_v2(num_classes=len(classes), pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def compute_class_weights(ds) -> torch.Tensor:
        # Support Dataset or Subset
        if hasattr(ds, "dataset") and hasattr(ds.dataset, "samples") and hasattr(ds, "indices"):
            labels = [ds.dataset.samples[i][1] for i in ds.indices]
        elif hasattr(ds, "samples"):
            labels = [lbl for _, lbl in ds.samples]
        else:
            # Fallback: iterate once (slow)
            labels = []
            for _, lbl in ds:
                labels.append(int(lbl))
        counts = Counter(labels)
        total = sum(counts.values())
        num_classes_local = len(classes)
        weights = [total / (num_classes_local * counts.get(i, 1)) for i in range(num_classes_local)]
        return torch.tensor(weights, dtype=torch.float32, device=device)

    if args.use_class_weights:
        class_weights = compute_class_weights(train_ds)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    no_improve = 0
    save_path = Path(args.out_dir) / "best_model_effnet.pt"
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            pred = outputs.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                pred = outputs.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else 0.0
        print(f"Epoch {epoch+1}/{args.epochs} - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_acc: {val_acc:.4f}")
        if val_acc > best_val + 1e-6:
            best_val = val_acc
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(save_path))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping after {epoch+1} epochs (no improvement for {args.patience} epochs)")
                break

    # Test
    if save_path.exists():
        print(f"Loading best weights from {save_path} for testing...")
        model.load_state_dict(torch.load(str(save_path), map_location=device))
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            pred = outputs.argmax(1).cpu().numpy()
            all_preds.append(pred)
            all_labels.append(labels.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    print(confusion_matrix(y_true, y_pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="processed data root with train/val/test")
    ap.add_argument("--out-dir", default="/tmp/kidney_runs")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--limit-samples", type=int, default=0, help="Limit samples per train split for quick runs")
    ap.add_argument("--use-class-weights", action="store_true")
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
