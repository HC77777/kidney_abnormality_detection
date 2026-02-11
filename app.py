import sys
from pathlib import Path

# Add src to path to reuse our modules
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
import plotly.graph_objects as go

# Import our new model definition
from model.efficientnet import build_efficientnet_v2

# --- Configuration ---
MODEL_PATH = Path("outputs/best_model_effnet.pt")
CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]
SIZE = 384 # High-Res

# --- Advanced Preprocessing Logic ---
def smart_crop_body(img_gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_gray
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    margin = 5
    h_img, w_img = img_gray.shape
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w_img, x + w + margin)
    y2 = min(h_img, y + h + margin)
    return img_gray[y1:y2, x1:x2]

def apply_clahe(img: np.ndarray, clip_limit=2.0) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(img)

def isotropic_resize(img: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = img.shape
    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def preprocess_debug(image_pil, use_clahe=True):
    img_gray = np.array(image_pil.convert("L"))
    img_cropped = smart_crop_body(img_gray)
    
    if use_clahe:
        img_clahe = apply_clahe(img_cropped)
    else:
        img_clahe = img_cropped
        
    img_final = isotropic_resize(img_clahe, size=SIZE)
    img_model_input = cv2.cvtColor(img_final, cv2.COLOR_GRAY2RGB)
    return img_gray, img_cropped, img_clahe, img_final, Image.fromarray(img_model_input)

# --- Model Logic ---
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = build_efficientnet_v2(num_classes=len(CLASSES), pretrained=False)
    if not MODEL_PATH.exists():
        return None
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def get_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def compute_gradcam_plusplus(model, img_tensor, target_class=None):
    target_layer = model.features[-1]
    gradients = []
    activations = []
    def forward_hook(module, inp, out):
        activations.append(out)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    model.zero_grad()
    logits = model(img_tensor)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
    score = logits[:, target_class]
    score.backward()
    h1.remove()
    h2.remove()
    grads = gradients[0].detach()
    acts = activations[0].detach()
    b, k, u, v = grads.shape
    grad_2 = grads.pow(2)
    grad_3 = grads.pow(3)
    sum_acts = acts.sum(dim=(2, 3), keepdim=True)
    alpha_num = grad_2
    alpha_denom = 2 * grad_2 + sum_acts * grad_3
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom
    pos_grads = F.relu(grads)
    weights = (alphas * pos_grads).sum(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze(0)
    cam = F.relu(cam)
    cam = cam.cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam, target_class, torch.softmax(logits, dim=1)[0]

def overlay_heatmap(original_pil, cam_map):
    img_cv = np.array(original_pil.convert("RGB"))
    h, w = img_cv.shape[:2]
    cam_resized = cv2.resize(cam_map, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.35 * heatmap + 0.65 * img_cv).astype(np.uint8)
    return overlay

# --- UI Components ---
def plot_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence %"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "darkblue"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plot_donut(probs, classes):
    # Filter small probabilities for cleaner chart
    labels = []
    values = []
    for p, c in zip(probs, classes):
        if p > 0.01: # Only show > 1%
            labels.append(c)
            values.append(p)
            
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])
    fig.update_layout(title_text="Probability Distribution", height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- Main App ---
st.set_page_config(page_title="Kidney CT AI", layout="wide", page_icon="🏥")

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #2C3E50;}
    .sub-header {font-size: 1.5rem; color: #34495E;}
    .card {padding: 20px; border-radius: 10px; background-color: #f8f9fa; margin-bottom: 20px;}
    .stProgress > div > div > div > div {background-color: #2E86C1;}
</style>
""", unsafe_allow_html=True)

mode = st.sidebar.radio("Mode", ["Diagnosis (Patient Study)", "Preprocessing Playground"])

if mode == "Preprocessing Playground":
    st.markdown('<p class="main-header">🛠️ Advanced Preprocessing Playground</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a Raw CT Image", type=["jpg", "png", "jpeg"])
    use_clahe = st.checkbox("Enable CLAHE (Contrast Enhancement)", value=True)
    
    if uploaded_file:
        raw_image = Image.open(uploaded_file)
        gray, cropped, clahe, final, model_input = preprocess_debug(raw_image, use_clahe=use_clahe)
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.image(gray, caption="1. Original", use_container_width=True)
        with col2: st.image(cropped, caption="2. Smart Crop", use_container_width=True)
        with col3: st.image(clahe, caption="3. Contrast", use_container_width=True)
        with col4: st.image(final, caption=f"4. Final ({SIZE}x{SIZE})", use_container_width=True)

elif mode == "Diagnosis (Patient Study)":
    st.markdown('<p class="main-header">🏥 AI Kidney Diagnosis</p>', unsafe_allow_html=True)
    st.write("Upload one or more CT slices. The AI will analyze each slice and aggregate findings.")
    
    uploaded_files = st.file_uploader("Upload Patient Scans", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        model = load_model()
        if model is None:
            st.error("Model not found! Training might still be in progress.")
            st.stop()
            
        tf = get_transform()
        
        results = []
        progress_text = "Analyzing scans..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, f in enumerate(uploaded_files):
            img = Image.open(f)
            _, _, _, _, processed_pil = preprocess_debug(img, use_clahe=True)
            img_tensor = tf(processed_pil).unsqueeze(0)
            
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                conf, pred_idx = torch.max(probs, dim=0)
            
            results.append({
                "filename": f.name,
                "probs": probs.numpy(),
                "conf": conf.item(),
                "pred_idx": pred_idx.item(),
                "pred_label": CLASSES[pred_idx.item()],
                "img_pil": processed_pil,
                "img_original": img,
                "original_file": f
            })
            my_bar.progress((i + 1) / len(uploaded_files), text=f"Processing {f.name}...")
            
        my_bar.empty()
        
        # Top-K Aggregation
        abnormal_score = lambda x: 1.0 - x["probs"][1] # Normal is index 1
        results.sort(key=abnormal_score, reverse=True)
        
        K = min(5, len(results))
        top_k_results = results[:K]
        
        avg_probs = np.mean([r["probs"] for r in top_k_results], axis=0)
        final_pred_idx = np.argmax(avg_probs)
        final_label = CLASSES[final_pred_idx]
        final_conf = avg_probs[final_pred_idx]
        
        # --- DASHBOARD LAYOUT ---
        st.divider()
        
        # Top Row: Diagnosis & Metrics
        c1, c2, c3 = st.columns([2, 1, 1])
        
        with c1:
            st.markdown(f"### Patient Diagnosis: **{final_label}**")
            if final_conf > 0.8:
                st.success(f"High Confidence ({final_conf:.1%})")
            elif final_conf > 0.6:
                st.warning(f"Moderate Confidence ({final_conf:.1%})")
            else:
                st.error(f"Low Confidence ({final_conf:.1%})")
            st.info(f"Analysis based on {len(uploaded_files)} slices. Diagnosis derived from Top {K} most suspicious findings.")

        with c2:
            st.plotly_chart(plot_gauge(final_conf), use_container_width=True)
            
        with c3:
            st.plotly_chart(plot_donut(avg_probs, CLASSES), use_container_width=True)
            
        # --- Filmstrip Gallery ---
        st.divider()
        st.markdown('<p class="sub-header">🔍 Evidence: Top Suspicious Slices</p>', unsafe_allow_html=True)
        
        for i, res in enumerate(top_k_results):
            # Grad-CAM
            img_tensor = tf(res["img_pil"]).unsqueeze(0)
            cam_map, _, _ = compute_gradcam_plusplus(model, img_tensor, target_class=res["pred_idx"])
            overlay = overlay_heatmap(res["img_pil"], cam_map)
            
            with st.container():
                c_img, c_cam, c_info = st.columns([1, 1, 2])
                with c_img:
                    st.image(res["img_original"], caption="Original Scan", use_container_width=True)
                with c_cam:
                    st.image(overlay, caption="AI Attention (Grad-CAM++)", use_container_width=True)
                with c_info:
                    st.markdown(f"#### Slice: {res['filename']}")
                    st.markdown(f"**Finding:** {res['pred_label']}")
                    st.markdown(f"**Confidence:** {res['conf']:.1%}")
                    st.write("Red areas indicate regions the AI found suspicious.")
            st.divider()
