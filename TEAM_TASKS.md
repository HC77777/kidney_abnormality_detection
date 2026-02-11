# Kidney CT Analysis Project - Team Task Division

**Project Goal:** Build an AI system to detect Cyst, Stone, and Tumor from 2D Kidney CT slices, aggregating results to a patient-level diagnosis.

**Core Tech Stack:** EfficientNetV2 (Model), Streamlit (UI), PyTorch (Framework).

---

## 🧑‍💻 Team Member 1: Data Engineer (Collection & Management)
**Focus:** Raw Data Handling & Pipeline Input.
*   **Task 1.1:** Source additional datasets (public repos like TCIA or Kaggle) specifically looking for **Hydronephrosis** cases to expand our class list.
*   **Task 1.2:** Write a script to parse raw **DICOM** files (standard medical format) and extract metadata (Patient ID, Slice Thickness).
*   **Task 1.3:** Implement a "Cleaner" script to remove non-axial scans or corrupted files before they reach the preprocessing stage.
*   **Task 1.4:** Maintain the `splits/` (train/val/test) CSVs, ensuring no patient leaks between training and testing sets.

## 🧑‍🔬 Team Member 2: Computer Vision Specialist (Preprocessing)
**Focus:** Image Quality & Artifact Removal.
*   **Task 2.1:** Refine the `smart_crop_body` function. Test it on 100 difficult images (e.g., obese patients, different scanners) to ensure it never cuts off the kidney.
*   **Task 2.2:** Experiment with **Windowing**. Instead of standard 0-255 normalization, implement specific Hounsfield Unit (HU) windows for kidneys (WL: 40, WW: 400).
*   **Task 2.3:** Implement **Test-Time Augmentation (TTA)**. During prediction, generate 3 versions of the image (rotated, flipped), predict on all, and average the result for higher accuracy.

## 🧠 Team Member 3: AI Researcher (Model Training)
**Focus:** The Neural Network.
*   **Task 3.1:** Monitor the **EfficientNetV2** training. Plot Loss and Accuracy curves. If it overfits, increase Dropout or Weight Decay.
*   **Task 3.2:** Benchmark against **ConvNeXt** or **ResNet50**. Prove that EfficientNetV2 is indeed the best choice.
*   **Task 3.3:** Implement **Label Smoothing**. This prevents the model from being "overconfident" and improves generalization on unseen data.
*   **Task 3.4:** Generate the **Confusion Matrix** and **ROC Curves** for the final report.

## ⚙️ Team Member 4: Logic & Backend Engineer (Aggregation)
**Focus:** From "Image Prediction" to "Patient Diagnosis".
*   **Task 4.1:** Implement the **Top-K Voting Logic**. (e.g., "If >5 slices are 'Tumor' with >90% confidence, flag patient as Tumor").
*   **Task 4.2:** Build an API (function) that accepts a **List of Images** (a full patient scan) and returns a JSON response with the diagnosis.
*   **Task 4.3:** Optimize inference speed. Ensure the model uses `torch.no_grad()` and batch processing so a 500-slice scan takes seconds, not minutes.

## 🎨 Team Member 5: Full Stack / Frontend (Deployment)
**Focus:** User Experience (The App).
*   **Task 5.1:** Upgrade `app.py`. Add support for **"Upload Folder"** or **"Upload Zip"** to handle full patient scans (not just single images).
*   **Task 5.2:** Build the **Results Dashboard**. Show the "Patient Diagnosis" in big text, followed by a gallery of the "Top Suspicious Slices" with their Grad-CAM heatmaps.
*   **Task 5.3:** Add a **"Download Report"** button that generates a PDF summary of the findings.
*   **Task 5.4:** Style the app (Custom CSS, Logo) to look professional and clinical.

