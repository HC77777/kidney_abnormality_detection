import torch
import torch.nn as nn
import torchvision

def build_efficientnet_v2(num_classes: int, pretrained: bool = True):
    """
    Builds an EfficientNetV2-S model.
    
    Args:
        num_classes: Number of output classes (e.g., 4 for Cyst, Normal, Stone, Tumor).
        pretrained: Whether to load ImageNet pretrained weights.
        
    Returns:
        model: The PyTorch model.
    """
    weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.efficientnet_v2_s(weights=weights)
    
    # Replace the classifier head
    # EfficientNet V2 has a 'classifier' block which is a Sequential containing Dropout and Linear
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

