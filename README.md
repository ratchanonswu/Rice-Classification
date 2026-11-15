# Rice Leaf Disease Classification: EfficientNetV2-S vs. ConvNeXtBase

A controlled empirical benchmark comparing **EfficientNetV2-S** (Lightweight Champion) and **ConvNeXtBase** (Modern ConvNet) for classifying rice leaf diseases, evaluated on a deduplicated, leakage-free dataset split with Explainable AI (**Grad-CAM**) visual attribution and a unified interactive web application.

---

## 📊 Benchmark Results (Clean Split v1)

Both models were evaluated on the exact same **570 held-out test images** under identical experimental protocols on a clean, group-stratified test split.

| Metric | EfficientNetV2-S (Light) | ConvNeXtBase (Deep) | Difference (Δ) | Winner |
|---|:---:|:---:|:---:|:---:|
| **Test Accuracy** | **92.81%** | 90.18% | +2.63% | **EfficientNetV2-S** |
| **Macro F1-Score** | **92.74%** | 90.17% | +2.57% | **EfficientNetV2-S** |
| **Macro Precision** | **92.81%** | 90.37% | +2.44% | **EfficientNetV2-S** |
| **Macro Recall** | **92.76%** | 90.12% | +2.64% | **EfficientNetV2-S** |
| **ROC-AUC (Macro OvR)** | **99.24%** | 98.92% | +0.32% | **EfficientNetV2-S** |
| **Model Checkpoint Size** | **83.04 MB** | 338.00 MB | −75.4% | **EfficientNetV2-S** |
| **Total Parameters** | **20,660,838** | 87,832,230 | −76.5% | **EfficientNetV2-S** |
| **Training Time (25 ep)** | **30.13 min** | 42.37 min | −28.9% | **EfficientNetV2-S** |
| **Avg Inference Latency (CPU)** | **77.8 ms** | 165.2 ms | −52.9% | **EfficientNetV2-S** |
| **Throughput (FPS on CPU)** | **12.9 fps** | 6.1 fps | +111.5% | **EfficientNetV2-S** |

> **Key Finding:** EfficientNetV2-S achieved superior generalization across all classification metrics (+2.63% Test Accuracy, +2.57% Macro F1) while requiring **4.25× fewer parameters** and **4.1× less storage** than ConvNeXtBase. Its fused MBConv blocks proved highly effective at capturing fine-grained foliar lesion boundaries.

---

## 🌾 The 6 Disease Classes

The dataset covers six critical rice foliage pathology categories:

| Index | Class Name | Pathogen / Description | EfficientNetV2-S F1 | ConvNeXtBase F1 |
|:---:|---|---|:---:|:---:|
| `0` | **Bacterial Leaf Blight** | Caused by *Xanthomonas oryzae*; marginal water-soaked lesions progressing to yellow-white stripes. | **89.47%** | 87.23% |
| `1` | **Brown Spot** | Caused by *Cochliobolus miyabeanus*; oval brown spots with yellow halos and dark necrotic centers. | **92.71%** | 90.53% |
| `2` | **Healthy Rice Leaf** | Clean, uninfected foliage displaying uniform green coloration (baseline control). | **97.49%** | 95.88% |
| `3` | **Leaf Blast** | Caused by *Magnaporthe oryzae*; spindle-shaped or diamond-shaped lesions with gray centers. | **89.89%** | 86.81% |
| `4` | **Leaf Scald** | Caused by *Microdochium oryzae*; zonate tan scalds beginning from leaf tips with alternating bands. | **91.67%** | 89.36% |
| `5` | **Sheath Blight** | Caused by *Rhizoctonia solani*; irregular greenish-gray water-soaked lesions with dark brown margins. | **95.24%** | 93.33% |

---

## 🔍 Explainable AI: Grad-CAM Feature Attribution

To ensure model predictions are based on clinically valid pathological features rather than background artifacts, both models are instrumented with **Gradient-weighted Class Activation Mapping (Grad-CAM)**:

- **Head-Propagation Method:** Instead of relying on internal intermediate hooks, the convolutional backbone's feature maps are watched directly through the classification head (`GlobalAveragePooling2D → Dense(6) Softmax`) using a `tf.GradientTape`. This guarantees reliable gradient flow directly from saved `.keras` checkpoints.
- **Architectural Attention Patterns:**
  - **EfficientNetV2-S:** Concentrates gradient activations sharply on pinpoint necrotic lesion margins, providing highly localized visual explanations.
  - **ConvNeXtBase:** With its large $7 \times 7$ depthwise kernels, pools contextual signals across broader sections of leaf tissue, providing complementary global leaf context.
- **Interactive Superimposition:** Normalized Grad-CAM heatmaps are colored using OpenCV's `COLORMAP_JET` and dynamically blended with the original leaf image ($\alpha=0.45$) as Base64 Data URLs for zero-latency browser rendering.

---

## 🛡️ Dataset Integrity & Leakage Prevention Pipeline

Data leakage from duplicate images across train/val/test splits artificially inflates model accuracy. A rigorous multi-stage audit was executed:

1. **Exact Duplicate Detection:** SHA-256 hashing identified 31 exact duplicate groups (62 redundant files).
2. **Deduplication Strategy:** Exactly 31 redundant copies were isolated, keeping one canonical instance per group.
3. **Near-Duplicate Audit:** Perceptual hashing (pHash with Hamming distance $\le 10$) flagged visually similar pairs with slight rotations or lighting shifts.
4. **Group-Stratified Splitting (Clean Split v1):**
   - **Total Clean Dataset:** 3,798 images
   - **Training Set (70%):** 2,658 images
   - **Validation Set (15%):** 570 images
   - **Test Set (15%):** 570 images (zero overlap with train or val)

---

## 📁 Final Deliverable Structure

```
FinalVersion/
├── models/
│   ├── efficientnetv2s.keras          # Champion model (92.81% acc, 83.0 MB, 20.7M params)
│   └── convnextbase.keras             # Modern ConvNet (90.18% acc, 338.0 MB, 87.8M params)
│
├── outputs/
│   ├── efficientnetv2s/               # Metrics, classification report, confusion matrix
│   ├── convnextbase/                  # Metrics, classification report, confusion matrix
│   └── class_images.json              # Pre-cached representative sample thumbnails
│
├── RiceVisionAI/                      # Complete Unified Web Application
│   ├── backend/
│   │   ├── main.py                    # FastAPI server (Inference + Grad-CAM + Static SPA)
│   │   └── requirements.txt           # Python dependencies
│   ├── frontend/
│   │   ├── src/                       # React 19 + TypeScript + Tailwind CSS UI
│   │   ├── dist/                      # Pre-compiled static production assets
│   │   └── package.json               # Frontend dependencies
│   ├── start_app.bat                  # Local app launcher
│   ├── start_backend.bat              # Standalone backend launcher (:8000)
│   └── start_frontend.bat             # Standalone frontend dev launcher (:5173)
│
├── notebooks/
│   └── Colab_RiceDisease_EfficientNetV2S_vs_ConvNeXtBase _Fix.ipynb # Complete training notebook
│
├── dataset_manifests/
│   ├── train.csv                      # Audited train split manifest (2,658 samples)
│   ├── validation.csv                 # Audited validation split manifest (570 samples)
│   ├── test.csv                       # Audited test split manifest (570 samples)
│   └── duplicate_groups.csv           # Deduplication audit manifest
│
├── start_app.bat                      # 1-Click launcher (starts server & opens browser)
└── README.md                          # Final project documentation (this file)
```

---

## 🚀 Interactive Web Application (RiceVision AI)

The web application provides an interactive decision-support interface and research dashboard.

### 1-Click Launch (Unified Server)

Double-click **`start_app.bat`** (or execute `.\start_app.bat` from PowerShell).

This automatically starts the FastAPI server and opens your browser at **`http://localhost:8000`**:
- **⚡ 1-Click Dual Mode Predictor (`/predict`):** Upload any leaf photo or choose from the built-in 6-class sample gallery to run **EfficientNetV2-S** and **ConvNeXtBase** side-by-side.
- **👁️ Grad-CAM Visualizer:** Compare side-by-side heatmaps with interactive view mode toggles (*Overlay Heatmap*, *Raw Heatmap*, *Original Leaf*) and an adjustable **Opacity Slider** (20% – 100%).
- **⚖️ Consensus Detection:** Displays high-confidence consensus when models agree or highlights differential diagnoses when they diverge.
- **📊 Model Comparison Dashboard (`/comparison`):** Head-to-head radar performance profiles, full $6 \times 6$ interactive confusion matrices, and parameter efficiency tables.
- **📖 Disease Class Reference (`/classes`):** Symptoms, high-resolution imagery, and agronomist-recommended intervention guidelines.

---

## ⚙️ Google Colab Training & Reproduction

To retrain or reproduce the benchmark in Google Colab:
1. Open [`notebooks/Colab_RiceDisease_EfficientNetV2S_vs_ConvNeXtBase _Fix.ipynb`](notebooks/Colab_RiceDisease_EfficientNetV2S_vs_ConvNeXtBase%20_Fix.ipynb) in Google Colab.
2. Select runtime: **Python 3 with T4 GPU**.
3. Mount Google Drive and verify dataset manifests under `dataset_manifests/`.
4. Execute cells sequentially:
   - Cell 1-10: Environment setup, random seed fixing (`seed=42`), and dataset preparation.
   - Cell 11-18: EfficientNetV2-S fine-tuning with cosine decay learning rate and early stopping.
   - Cell 19-26: ConvNeXtBase fine-tuning under identical optimizer and loss protocols.
   - Cell 27-34: Standardized latency profiling and Grad-CAM generation.

---

## ⚠️ Academic & Practical Disclaimer

This system is an **educational research prototype** developed for comparative computer vision research. Model prediction scores and Grad-CAM feature heatmaps are deep learning research outputs intended for decision-support and should be verified with qualified agricultural extension officers or plant pathologists before implementing commercial crop treatments.
