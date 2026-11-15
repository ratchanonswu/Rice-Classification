# Rice Leaf Disease Classification: EfficientNetV2-S vs. Vision Transformer (ViT-B/16)

A controlled empirical benchmark comparing **EfficientNetV2-S** (Lightweight CNN) and **ViT-B/16** (Vision Transformer) for classifying rice leaf diseases, evaluated on a SHA-256 deduplicated, leakage-free dataset split with dual Explainable AI (**Grad-CAM++ & Attention Saliency**) visual attribution and a unified interactive web application.

---

## 📊 Benchmark Results (Clean Split)

Both models were evaluated on the exact same **570 held-out test images** under identical experimental protocols on a clean, leakage-free test split.

| Metric | EfficientNetV2-S (CNN) | ViT-B/16 (Transformer) | Difference (Δ) | Winner |
|---|:---:|:---:|:---:|:---:|
| **Test Accuracy** | 95.26% | **97.72%** | +2.46% | **ViT-B/16** |
| **Macro F1-Score** | 95.21% | **97.71%** | +2.50% | **ViT-B/16** |
| **Macro Precision** | 95.23% | **97.80%** | +2.57% | **ViT-B/16** |
| **Macro Recall** | 95.22% | **97.70%** | +2.48% | **ViT-B/16** |
| **ROC-AUC (Macro OvR)** | 99.61% | **99.74%** | +0.13% | **ViT-B/16** |
| **Model Checkpoint Size** | **83.04 MB** | 982.52 MB | −91.5% | **EfficientNetV2-S** |
| **Total Parameters** | **20,660,838** | 85,803,270 | −75.9% | **EfficientNetV2-S** |
| **Training Time** | 9.98 min | **4.75 min** | −52.4% | **ViT-B/16** |
| **Avg Inference Latency (GPU)** | 14.88 ms | **7.64 ms** | −48.7% | **ViT-B/16** |
| **Throughput (FPS on GPU)** | 67.2 fps | **130.8 fps** | +94.6% | **ViT-B/16** |

> **Key Finding:** **ViT-B/16** achieved the highest generalization across every classification metric (**97.72% Test Accuracy**, **97.71% Macro F1**, **130.8 FPS** on GPU) by using global self-attention to eliminate background shortcut bias. Meanwhile, **EfficientNetV2-S** remains **11.8× lighter** (**83.04 MB** vs. **982.52 MB**) while still achieving **95.26% Test Accuracy**, making it ideal for RAM-constrained deployments.

---

## 🌾 The 6 Disease Classes

The dataset covers six critical rice foliage pathology categories:

| Index | Class Name | Pathogen / Description | EfficientNetV2-S F1 | ViT-B/16 F1 |
|:---:|---|---|:---:|:---:|
| `0` | **Bacterial Leaf Blight** | Caused by *Xanthomonas oryzae*; marginal water-soaked lesions progressing to yellow-white stripes. | 94.79% | **97.38%** |
| `1` | **Brown Spot** | Caused by *Cochliobolus miyabeanus*; oval brown spots with yellow halos and dark necrotic centers. | 95.29% | **97.94%** |
| `2` | **Healthy Rice Leaf** | Clean, uninfected foliage displaying uniform green coloration (baseline control). | 98.99% | **99.49%** |
| `3` | **Leaf Blast** | Caused by *Magnaporthe oryzae*; spindle-shaped or diamond-shaped lesions with gray centers. | 92.31% | **96.67%** |
| `4` | **Leaf Scald** | Caused by *Microdochium oryzae*; zonate tan scalds beginning from leaf tips with alternating bands. | 93.55% | **95.88%** |
| `5` | **Sheath Blight** | Caused by *Rhizoctonia solani*; irregular greenish-gray water-soaked lesions with dark brown margins. | 96.34% | **98.92%** |

---

## 🔍 Explainable AI: Grad-CAM++ & ViT Attention Saliency

To ensure model predictions rely on clinically valid pathological features rather than background artifacts (shortcut learning), both architectures are instrumented with visual attribution:

- **EfficientNetV2-S (Grad-CAM++ / Head-Propagation Grad-CAM):**
  - In the Colab research notebook (`compute_gradcam_plusplus`), higher-order derivatives (1st, 2nd, and 3rd-order gradients via nested `tf.GradientTape`) weight feature maps to isolate sharp, multi-lesion borders.
  - In the real-time FastAPI backend (`_compute_gradcam_heatmap`), Head-Propagation Grad-CAM watches the backbone feature maps directly through the classification head for low-latency inference.
- **ViT-B/16 (14×14 Patch Attention Saliency):**
  - Because KerasHub `ViTImageClassifier` splits `224×224` images into `16×16` patches (`14×14 = 196` tokens) and pools via a 1D CLS token, `_compute_vit_saliency_heatmap` computes input-gradient saliency ($|\partial \text{score} / \partial x|$), average-pools into a `14×14` patch grid (`ksize=16, strides=16`), and clips outliers at the **99th percentile**.
- **Interactive Superimposition:** Normalized heatmaps are colored using OpenCV's `COLORMAP_JET` and blended with the original leaf image ($\alpha = 0.45$) as Base64 Data URLs for zero-latency browser rendering.

---

## 🛡️ Dataset Integrity & Leakage Prevention Pipeline

Data leakage from duplicate images across train/val/test splits artificially inflates model accuracy. A multi-stage audit was executed and recorded in `dataset_manifests/`:

1. **Exact Duplicate Detection:** SHA-256 hashing identified 31 exact duplicate groups (62 redundant files).
2. **Deduplication Strategy:** Exactly 31 redundant copies were isolated, keeping one canonical instance per group.
3. **Near-Duplicate Audit:** Perceptual hashing (pHash with Hamming distance $\le 10$) flagged visually similar pairs with slight rotations or lighting shifts.
4. **Group-Stratified Splitting (Clean Split v1):**
   - **Total Clean Dataset:** 3,798 images
   - **Training Set (75%):** 2,847 images
   - **Validation Set (10%):** 381 images
   - **Test Set (15%):** 570 images (zero overlap with train or val)

---

## 📁 Project Structure

```
Rice Classification/
├── models/
│   ├── efficientnetv2s.keras          # Lightweight CNN (95.26% acc, 83.04 MB, 20.66M params)
│   └── vit_base.keras                 # Vision Transformer (97.72% acc, 982.52 MB, 85.80M params)
│
├── outputs/
│   ├── efficientnetv2s/               # Metrics, classification report, confusion matrix, curves
│   ├── vit_base/                      # Metrics, classification report, confusion matrix, curves
│   ├── sample_results.json            # Pre-computed XAI sample predictions for instant demo
│   └── class_images.json              # Pre-cached representative sample thumbnails
│
├── RiceVisionAI/                      # Complete Unified Web Application
│   ├── backend/
│   │   ├── main.py                    # FastAPI server (Inference + XAI + Static SPA)
│   │   └── requirements.txt           # Python dependencies (incl. keras-hub)
│   ├── frontend/
│   │   ├── src/                       # React 19 + TypeScript + Tailwind CSS UI
│   │   ├── dist/                      # Pre-compiled static production assets
│   │   └── package.json               # Frontend dependencies
│   ├── start_app.bat                  # Local app launcher
│   ├── start_backend.bat              # Standalone backend launcher (:8000)
│   └── start_frontend.bat             # Standalone frontend dev launcher (:5173)
│
├── notebooks/
│   └── Colab_RiceDisease_EfficientNetV2S_vs_ViTB16.ipynb # Complete training & XAI notebook
│
├── dataset_manifests/
│   ├── train.csv                      # Audited train split manifest (2,847 samples)
│   ├── validation.csv                 # Audited validation split manifest (381 samples)
│   ├── test.csv                       # Audited test split manifest (570 samples)
│   ├── all_images_audit.csv           # Full 3,829-file SHA-256 audit log
│   └── duplicate_groups.csv           # Exact & near-duplicate cluster manifest
│
├── generate_sample_results.py         # Pre-computes XAI sample gallery JSON
├── start_app.bat                      # 1-Click launcher (starts server & opens browser)
└── README.md                          # Project documentation (this file)
```

---

## 🚀 Interactive Web Application (RiceVision AI)

The web application provides an interactive decision-support interface and research dashboard.

### 1-Click Launch (Unified Server)

Double-click **`start_app.bat`** (or execute `.\start_app.bat` from PowerShell).

This automatically starts the FastAPI server and opens your browser at **`http://localhost:8000`**:
- **⚡ 1-Click Dual Mode Predictor (`/predict`):** Upload any leaf photo or choose from the built-in 6-class sample gallery to run **EfficientNetV2-S** and **ViT-B/16** side-by-side (or test live in the [Google Colab Demo](https://colab.research.google.com/drive/1PdU8uo0_waMQJ6sf2Qq-6Uh0GeEk-qOy?usp=sharing)).
- **👁️ Dual XAI Visualizer:** Compare **Grad-CAM** (EfficientNetV2-S) and **Attention Saliency** (ViT-B/16) heatmaps with interactive view mode toggles (*Overlay Heatmap*, *Raw Heatmap*, *Original Leaf*) and an adjustable **Opacity Slider** (20% – 100%).
- **⚖️ Consensus Detection:** Displays high-confidence consensus when models agree or highlights differential diagnoses when they diverge.
- **📊 Model Comparison Dashboard (`/comparison`):** Head-to-head radar performance profiles, full $6 \times 6$ interactive confusion matrices, and parameter efficiency tables.
- **📖 Disease Class Reference (`/classes`):** Symptoms, high-resolution imagery, and agronomist-recommended intervention guidelines.

---

## ⚙️ Google Colab Training & Reproduction

To retrain or reproduce the benchmark in Google Colab:
1. Open [`notebooks/Colab_RiceDisease_EfficientNetV2S_vs_ViTB16.ipynb`](notebooks/Colab_RiceDisease_EfficientNetV2S_vs_ViTB16.ipynb) in Google Colab.
2. Select runtime: **Python 3 with A100 / L4 GPU**.
3. Mount Google Drive and verify dataset manifests under `dataset_manifests/`.
4. Execute cells sequentially:
   - Setup & Data Audit: Environment setup (`keras-hub`), random seed fixing (`seed=42`), SHA-256 deduplication & zero-leakage verification.
   - EfficientNetV2-S Training: Two-stage transfer learning (frozen head + full fine-tuning).
   - ViT-B/16 Training: Fine-tuning `vit_base_patch16_224_imagenet` with AdamW / ReduceLROnPlateau.
   - Evaluation & XAI: Standardized latency profiling, head-to-head comparison table, and Grad-CAM++ vs. ViT Attention Saliency visualization.

---

## ⚠️ Academic & Practical Disclaimer

This system is an **educational research prototype** developed for comparative computer vision research. Model prediction scores and XAI heatmaps are deep learning research outputs intended for decision-support and should be verified with qualified agricultural extension officers or plant pathologists before implementing commercial crop treatments.
