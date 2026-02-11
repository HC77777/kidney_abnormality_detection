import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as T
from PIL import Image


CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]


def build_model(num_classes: int):
    model = torchvision.models.densenet121(weights=None)
    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, num_classes)
    return model


def get_transform(size: int = 224):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())
        score = logits[:, target_class]
        score.backward()

        gradients = self.gradients  # [B, C, H, W]
        activations = self.activations  # [B, C, H, W]
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = torch.sum(weights * activations, dim=1)  # [B, H, W]
        cam = torch.relu(cam)
        cam = cam[0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_cam_on_image(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img_rgb).astype(np.uint8)
    return overlay


def main():
    ap = argparse.ArgumentParser(description="Generate Grad-CAM overlays for DenseNet-121")
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--class", dest="cls", default="Normal", choices=CLASSES)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    model = build_model(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device).eval()

    # For DenseNet-121, last conv features are in features.denseblock4
    target_layer = model.features.denseblock4
    cam_gen = GradCAM(model, target_layer)

    tf = get_transform(args.size)
    cls = args.cls
    src_dir = Path(args.processed_root) / "test" / cls
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(src_dir.glob("*.jpg")):
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = tf(img_pil).unsqueeze(0).to(device)
        cam = cam_gen.generate(img_tensor)
        img_rgb = np.array(img_pil)
        overlay = overlay_cam_on_image(img_rgb, cam)
        out_path = out_dir / f"{img_path.stem}_gradcam.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        count += 1
        if count >= args.num:
            break
    print(f"Saved {count} Grad-CAM overlays to {out_dir}")


if __name__ == "__main__":
    main()


