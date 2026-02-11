import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch import nn
import torch.nn.functional as F

from model.efficientnet import build_efficientnet_v2

CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]


def get_transform(size: int = 224):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class GradCAMPlusPlus:
    """
    Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks
    Paper: https://arxiv.org/abs/1710.11063
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

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
            
        # Score for the target class
        score = logits[:, target_class]
        
        # Backward pass
        score.backward()

        # 1. Get Gradients and Activations
        # shape: [1, C, H, W]
        grads = self.gradients
        acts = self.activations
        
        # 2. Compute Alpha coefficients (Grad-CAM++ formula)
        # alpha_kc^ij = ... (complex formula involving 2nd and 3rd derivatives)
        # Since we can't easily compute 2nd/3rd derivatives in a simple backward hook without create_graph=True,
        # we use the standard closed-form approximation for Grad-CAM++ from gradients.
        
        # Detailed implementation:
        b, k, u, v = grads.shape
        
        grad_2 = grads.pow(2)
        grad_3 = grads.pow(3)
        
        # Sum over H, W
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        
        # Alpha numerators and denominators
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_acts * grad_3
        # Avoid div by zero
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        
        alphas = alpha_num / alpha_denom
        
        # 3. Compute Weights w_k^c
        # ReLU(gradients) because we only care about positive influence for the class
        pos_grads = F.relu(grads)
        
        weights = (alphas * pos_grads).sum(dim=(2, 3), keepdim=True)
        
        # 4. Compute Weighted Combination
        cam = (weights * acts).sum(dim=1)
        
        # 5. ReLU and Normalize
        cam = F.relu(cam)
        cam = cam[0].cpu().detach().numpy()
        
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
            
        return cam


def overlay_cam_on_image(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * img_rgb).astype(np.uint8)
    return overlay


def main():
    ap = argparse.ArgumentParser(description="Generate Grad-CAM++ overlays")
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--class", dest="cls", default="Tumor", choices=CLASSES)
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    model = build_efficientnet_v2(num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device).eval()

    # EfficientNetV2 Last Conv Block
    target_layer = model.features[-1]
    cam_gen = GradCAMPlusPlus(model, target_layer)

    tf = get_transform(args.size)
    cls = args.cls
    src_dir = Path(args.processed_root) / "test" / cls
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(src_dir.glob("*.jpg")):
        if count >= args.num:
            break
            
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = tf(img_pil).unsqueeze(0).to(device)
        
        cam = cam_gen.generate(img_tensor)
        
        img_rgb = np.array(img_pil)
        overlay = overlay_cam_on_image(img_rgb, cam)
        
        out_path = out_dir / f"{img_path.stem}_gradcam++.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        count += 1
        
    print(f"Saved {count} Grad-CAM++ overlays to {out_dir}")


if __name__ == "__main__":
    main()
