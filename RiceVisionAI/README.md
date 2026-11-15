# RiceVision AI

A research dashboard and live prediction demo for **Rice Leaf Disease Classification** comparing **EfficientNetV2-S** (Lightweight CNN) and **ViT-B/16** (Vision Transformer) on a SHA-256 deduplicated, leakage-free dataset with Explainable AI (**Grad-CAM** & **Attention Saliency**).

## Overview

This application was built as part of a university research project. It provides:

- **Overview dashboard** – dataset statistics, model summary, and feature highlights  
- **Model Comparison** – side-by-side metrics, per-class F1 chart, confusion matrices, and efficiency table  
- **Disease Classes** – visual browser of all six disease classes with descriptions, symptoms, and management guidelines  
- **Image Prediction** – upload a rice leaf image (or select pre-computed samples) and receive real-time predictions + XAI heatmaps from either or both models  
- **Methodology** – full pipeline documentation from SHA-256 deduplication to XAI & deployment  

## Project Structure

```
RiceVisionAI/
├── backend/
│   ├── main.py              ← FastAPI application (Inference + XAI + Static SPA)
│   └── requirements.txt     ← Python dependencies (incl. keras-hub)
├── frontend/
│   ├── src/
│   │   ├── pages/           ← Overview, Comparison, Classes, Predict, Methodology
│   │   ├── components/      ← Navbar, UI primitives
│   │   ├── api.ts           ← API client
│   │   ├── types.ts         ← TypeScript types
│   │   ├── App.tsx          ← Router
│   │   └── main.tsx         ← Entry point
│   ├── dist/                ← Compiled static production bundle
│   ├── index.html
│   └── package.json
├── start_backend.bat        ← Start FastAPI (Windows)
├── start_frontend.bat       ← Start Vite dev server (Windows)
└── README.md
```

## Requirements

### Backend
- Python 3.11+ with dependencies from `RiceVisionAI/backend/requirements.txt` (TensorFlow, Keras 3, `keras-hub>=0.17.0`, FastAPI, uvicorn, OpenCV, Pillow, NumPy)
- Trained model checkpoints in `models/`:
  - `models/efficientnetv2s.keras`
  - `models/vit_base.keras`

### Frontend
- Node.js 18+ and npm

## Installation

### 1. Install backend dependencies

```powershell
pip install -r RiceVisionAI\backend\requirements.txt
```

### 2. Install frontend dependencies

```powershell
cd RiceVisionAI\frontend
npm install
```

## Running the Application

### 1-Click Unified Server (Recommended)

From the project root, run `.\start_app.bat`, or start backend and frontend separately in two terminals:

### Terminal 1 – Start the backend

```powershell
cd RiceVisionAI
.\start_backend.bat
```

The API will be available at: `http://localhost:8000`  
Interactive API docs: `http://localhost:8000/docs`

### Terminal 2 – Start the frontend (dev mode)

```powershell
cd RiceVisionAI
.\start_frontend.bat
```

Open your browser at: **http://localhost:5173**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Backend and model availability check |
| GET | `/api/classes` | Class names, descriptions, representative images |
| GET | `/api/models` | Model metadata, metrics, efficiency data |
| GET | `/api/metrics` | Per-class metrics and confusion matrices |
| GET | `/api/sample-results` | Pre-computed 6-class sample predictions & XAI heatmaps |
| POST | `/api/predict?model=efficientnetv2s` | Predict + Grad-CAM from uploaded image |
| POST | `/api/predict?model=vit_base` | Predict + Attention Saliency from uploaded image |
| POST | `/api/predict?model=both` | Dual-model consensus prediction + side-by-side XAI |

## Models Used

| Model | Accuracy | Macro F1 | ROC-AUC | Size | Avg Inference (GPU) |
|-------|----------|----------|---------|------|---------------------|
| EfficientNetV2-S | 95.26% | 95.21% | 99.61% | 83.04 MB | 14.88 ms (67.2 FPS) |
| ViT-B/16 | 97.72% | 97.71% | 99.74% | 982.52 MB | 7.64 ms (130.8 FPS) |

Evaluated on 570 test images (SHA-256 deduplicated, zero-leakage split).

## Notes

- **First prediction may be slow** — models are loaded lazily on the first request. Subsequent predictions are faster.
- Images are processed in-memory. File system paths are never exposed to the browser.
- This is an **educational research prototype**, not a certified agricultural diagnostic system.
- Preprocessing & XAI:
  - **EfficientNetV2-S:** Internal Keras preprocessing (`0–255` input) + Head-Propagation Grad-CAM (`_compute_gradcam_heatmap`).
  - **ViT-B/16:** Rescaled `[-1, 1]` input + 14×14 Patch Attention Saliency (`_compute_vit_saliency_heatmap`, 99th percentile clipping).
