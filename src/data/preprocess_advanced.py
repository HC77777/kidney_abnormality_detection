import cv2
import numpy as np
from PIL import Image

def smart_crop_body(img_gray: np.ndarray) -> np.ndarray:
    """
    Finds the body in the CT slice (largest foreground object) 
    and crops exactly to its bounding box.
    Removes black scanning table and background air.
    """
    # 1. Threshold to find body (foreground) vs air (background)
    # Use Otsu's thresholding which automatically finds the best split value
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img_gray # Fallback: return original if no body found
        
    # 3. Find largest contour (the patient's body)
    c = max(contours, key=cv2.contourArea)
    
    # 4. Get bounding box
    x, y, w, h = cv2.boundingRect(c)
    
    # 5. Crop (with a tiny safety margin)
    margin = 5
    h_img, w_img = img_gray.shape
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)
    
    return img_gray[y1:y2, x1:x2]

def isotropic_resize(img: np.ndarray, size: int = 224) -> np.ndarray:
    """
    Resizes the image to (size, size) WITHOUT stretching/distortion.
    Adds black padding (letterboxing) to keep the aspect ratio.
    Crucial for stones/cysts where 'roundness' is a diagnostic feature.
    """
    h, w = img.shape
    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Create blank canvas
    canvas = np.zeros((size, size), dtype=np.uint8)
    
    # Center the image
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Enhances local details (texture inside kidney) without amplifying noise too much.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def preprocess_pipeline(image_path: str, target_size: int = 224) -> np.ndarray:
    """
    The Master Function: Raw File -> AI-Ready Tensor
    """
    # 1. Read as Grayscale (CT is fundamentally grayscale)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    # 2. Smart Crop (Remove background air/table)
    img = smart_crop_body(img)
    
    # 3. Enhance Contrast (CLAHE)
    # We do this BEFORE resize to preserve texture information
    img = apply_clahe(img)
    
    # 4. Isotropic Resize (Pad to square)
    img = isotropic_resize(img, size=target_size)
    
    # 5. Stack to 3 Channels (for EfficientNet/Transfer Learning compatibility)
    # Even though CT is gray, pretrained models expect RGB input.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return img_rgb

if __name__ == "__main__":
    # Quick Test
    import sys
    if len(sys.argv) > 1:
        res = preprocess_pipeline(sys.argv[1])
        cv2.imwrite("test_preprocessed_advanced.jpg", res)
        print("Saved test_preprocessed_advanced.jpg")

