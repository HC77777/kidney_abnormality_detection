# Kidney Abnormality Detection using CT Scans

This project implements an advanced deep learning pipeline to detect and classify kidney abnormalities (Normal, Stone, Cyst, Tumor) from CT scan images. It features a high-accuracy EfficientNetV2-S model, advanced image preprocessing (Smart Body Cropping, CLAHE, Isotropic Resize), study-level diagnostic aggregation using Top-K predictions, and Grad-CAM++ based Explainable AI (XAI) visualizations.

## Features

- **Advanced Preprocessing Pipeline**: Ensures uniform, high-contrast, artifact-free ROI extraction.
- **Top-Notch Deep Learning Model**: Utilizes `EfficientNetV2-S` fine-tuned to achieve near-perfect multi-class classification accuracy at high resolution (384x384).
- **Explainable AI (XAI)**: Generates Grad-CAM++ heatmaps to highlight suspicious areas on the kidney scans, allowing medical professionals to trust the model.
- **Study-Level Aggregation**: Analyzes multiple slices from a single patient's study and uses Top-K voting to provide an overall diagnosis, preventing normal slices from diluting positive signals.
- **Interactive Web App**: A Streamlit frontend for clinicians to upload slices, view diagnoses, understand model confidence with Plotly charts, and explore the preprocessing steps visually.

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HC77777/kidney_abnormality_detection.git
   cd kidney_abnormality_detection
   ```

2. **Set up a Python Virtual Environment** (optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Web App

The repository includes the fully trained `best_model_effnet.pt` (78 MB) inside the `outputs/` folder. You do NOT need to download the 4.7 GB dataset to run the prediction web app!

```bash
streamlit run app.py
```
Then, open the provided local URL in your browser.

## Training the Model (Optional)

If you wish to re-train the model from scratch, you will need the dataset.

1. **Download Dataset from Kaggle**:
   - Ensure you have your `kaggle.json` configured in `~/.kaggle/`.
   - Run the download command:
     ```bash
     kaggle datasets download -d nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone
     ```
   - Extract it into the `data/` directory (e.g., `data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone`).

2. **Preprocess and Split the Data**:
   - The data scripts inside `src/data/` handle deduplication, duplicate removal, pseudo-study splitting, and advanced processing (Smart crop, CLAHE, Resize).
   - Execute the scripts sequentially as they were structured in the repository (e.g. `python src/data/run_advanced_preprocess.py`).

3. **Run Training**:
   ```bash
   python src/train.py
   ```
   The best model weights will be saved to `outputs/best_model_effnet.pt`.

## Team Contributions

This repository contains the full end-to-end code logic. Please see `TEAM_TASKS.md` for a suggested 5-member team division strategy.
