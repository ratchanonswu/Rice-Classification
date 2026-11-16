"""
RiceVision AI – FastAPI Backend
Rice Leaf Disease Classification Research Dashboard & Explainable AI (Grad-CAM & ViT Saliency)
Dual Model Benchmark: EfficientNetV2-S vs Vision Transformer (ViT-B/16)
"""

import json
import os
import time
import base64
import io
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import keras_hub

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ─── Path constants ──────────────────────────────────────────────────────────

# Base paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent.parent.parent  # c:\Gitwork\Rice diseases

# Allow env-var overrides for containerised / cloud deployments
# (HuggingFace Spaces, Docker, Render, Railway, …)
_default_finals = BACKEND_DIR.parent.parent
if not (_default_finals / "models").exists():
    if (PROJECT_ROOT / "finals").exists():
        _default_finals = PROJECT_ROOT / "finals"
    elif (PROJECT_ROOT / "FinalVersion").exists():
        _default_finals = PROJECT_ROOT / "FinalVersion"

MODELS_DIR   = Path(os.getenv("MODELS_DIR",   str(_default_finals / "models")))
OUTPUTS_DIR  = Path(os.getenv("OUTPUTS_DIR",  str(_default_finals / "outputs")))
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", str(BACKEND_DIR.parent / "frontend" / "dist")))

DATASET_DIR = PROJECT_ROOT / "Rice Diseases Classification" / "ResNet50" / "dataset"

EFFIENTNET_CHECKPOINT = MODELS_DIR / "efficientnetv2s.keras"
VIT_CHECKPOINT        = MODELS_DIR / "vit_base.keras"

CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight",
]

# When set, only this model is loaded (e.g. "efficientnetv2s" on memory-limited deployments)
SINGLE_MODEL = os.getenv("SINGLE_MODEL", "")

# ─── App Initialization ──────────────────────────────────────────────────────

app = FastAPI(
    title="RiceVision AI",
    description="Rice Leaf Disease Classification – EfficientNetV2-S vs ViT-B/16 with Explainable AI",
    version="2.0.0",
)

# CORS configuration
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Cache ─────────────────────────────────────────────────────────────

_models: Dict[str, Any] = {}
_class_images: Optional[Dict[str, Optional[str]]] = None


def _get_class_names() -> List[str]:
    return CLASS_NAMES


def _load_model(name: str):
    """Load and cache a model by name ('efficientnetv2s' or 'vit_base')."""
    if name in _models:
        return _models[name]

    if name == "efficientnetv2s":
        path = EFFIENTNET_CHECKPOINT
    elif name == "vit_base":
        path = VIT_CHECKPOINT
    else:
        raise ValueError(f"Unknown model name: {name}")

    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")

    model = tf.keras.models.load_model(str(path))
    _models[name] = model
    return model


def _compute_gradcam_heatmap(img_tensor: tf.Tensor, model: Any, pred_index: Optional[int] = None) -> np.ndarray:
    """
    Computes a normalized Grad-CAM++ heatmap (Chattopadhay et al., 2018) for an input tensor
    (shape: 1, 224, 224, 3) using pre-softmax logits via head-propagation for EfficientNetV2-S.
    """
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
    backbone = model.layers[1]
    head_layers = model.layers[2:]

    with tf.GradientTape() as tape:
        features = backbone(img_tensor, training=False)
        tape.watch(features)

        x = features
        for layer in head_layers[:-1]:
            x = layer(x, training=False)
        last_layer = head_layers[-1]
        if hasattr(last_layer, "kernel") and hasattr(last_layer, "bias"):
            logits = tf.matmul(x, last_layer.kernel) + last_layer.bias
        else:
            logits = last_layer(x, training=False)

        if pred_index is None:
            pred_index = int(tf.argmax(logits[0]))
        score = logits[:, pred_index]

    grads = tape.gradient(score, features)
    if grads is None:
        raise ValueError("Gradients could not be computed for Grad-CAM++.")

    # Chattopadhay et al. (2018) closed-form Grad-CAM++ pixel-wise alpha weighting
    grads_pos = tf.maximum(grads[0], 0.0)
    grads_pow2 = tf.square(grads_pos)
    grads_pow3 = grads_pow2 * grads_pos
    sum_activations = tf.reduce_sum(features[0], axis=(0, 1), keepdims=True)
    alpha_denom = 2.0 * grads_pow2 + sum_activations * grads_pow3 + 1e-10
    alphas = tf.where(grads[0] > 0.0, grads_pow2 / alpha_denom, 0.0)

    weights = tf.reduce_sum(alphas * grads_pos, axis=(0, 1))
    cam = tf.maximum(tf.reduce_sum(weights * features[0], axis=-1), 0.0)
    max_val = tf.math.reduce_max(cam)
    if max_val > 0:
        cam = cam / max_val
    return cam.numpy()


def _compute_vit_saliency_heatmap(img_tensor: tf.Tensor, vit_model: Any) -> np.ndarray:
    """
    Computes gradient saliency attention map for ViT-B/16 pooled into 14x14 patches.
    """
    x = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        preds = vit_model(x, training=False)
        pred_idx = tf.argmax(preds[0])
        score = preds[0, pred_idx]
    grads = tape.gradient(score, x)
    sal = tf.reduce_sum(tf.abs(grads), axis=-1, keepdims=True)
    grid = tf.nn.avg_pool2d(sal, ksize=16, strides=16, padding="VALID")
    grid = grid[0, :, :, 0].numpy()
    grid = np.clip(grid, 0, np.percentile(grid, 99))
    if grid.max() > 0:
        grid = grid / grid.max()
    return grid


def _overlay_gradcam(original_img_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45):
    """Overlays Grad-CAM or Saliency heatmap onto original PIL image and generates Base64 Data URLs."""
    orig_np = np.array(original_img_pil.convert("RGB"))
    h, w, _ = orig_np.shape

    # Resize heatmap to match original image resolution
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed = np.uint8(orig_np * (1.0 - alpha) + heatmap_color * alpha)

    # Base64 encoded overlay
    buf_overlay = io.BytesIO()
    Image.fromarray(superimposed).save(buf_overlay, format="JPEG", quality=88)
    overlay_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf_overlay.getvalue()).decode('utf-8')}"

    # Base64 encoded raw heatmap
    buf_raw = io.BytesIO()
    Image.fromarray(heatmap_color).save(buf_raw, format="JPEG", quality=88)
    raw_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf_raw.getvalue()).decode('utf-8')}"

    return overlay_b64, raw_b64


def _run_single_inference(img: Image.Image, model_name: str) -> Dict[str, Any]:
    """Execute prediction and XAI generation for a single model."""
    model = _load_model(model_name)
    class_names = _get_class_names()

    # Preprocess image
    img_resized = img.convert("RGB").resize((224, 224))
    arr = np.array(img_resized, dtype=np.float32)

    if model_name == "vit_base":
        arr = (arr / 127.5) - 1.0
        img_tensor = tf.convert_to_tensor(np.expand_dims(arr, axis=0), dtype=tf.float32)
        t0 = time.perf_counter()
        preds = model(img_tensor, training=False).numpy()
        inference_ms = round((time.perf_counter() - t0) * 1000, 1)
        heatmap = _compute_vit_saliency_heatmap(img_tensor, model)
        model_label = "ViT-B/16 (Vision Transformer)"
    else:
        img_tensor = tf.convert_to_tensor(np.expand_dims(arr, axis=0), dtype=tf.float32)
        t0 = time.perf_counter()
        preds = model.predict(img_tensor, verbose=0)
        inference_ms = round((time.perf_counter() - t0) * 1000, 1)
        top_idx_eff = int(np.argmax(preds[0]))
        heatmap = _compute_gradcam_heatmap(img_tensor, model, top_idx_eff)
        model_label = "EfficientNetV2-S"

    scores = preds[0].tolist()
    top_idx = int(np.argmax(scores))

    overlay_b64, raw_b64 = _overlay_gradcam(img, heatmap, alpha=0.45)

    return {
        "model": model_name,
        "model_label": model_label,
        "predicted_class": class_names[top_idx],
        "predicted_index": top_idx,
        "confidence": round(float(scores[top_idx]), 4),
        "scores": {class_names[i]: round(float(scores[i]), 4) for i in range(len(class_names))},
        "inference_ms": inference_ms,
        "gradcam_overlay": overlay_b64,
        "gradcam_raw": raw_b64,
    }


def _get_class_images() -> Dict[str, Optional[str]]:
    """Return a dict mapping class name → base64-encoded thumbnail."""
    global _class_images
    if _class_images is not None:
        return _class_images

    cached_json = OUTPUTS_DIR / "class_images.json"
    if cached_json.exists():
        try:
            with open(cached_json, "r", encoding="utf-8") as f:
                _class_images = json.load(f)
                return _class_images
        except Exception:
            pass

    result: Dict[str, Optional[str]] = {}
    for cls in CLASS_NAMES:
        class_dir = DATASET_DIR / cls
        if not class_dir.exists():
            result[cls] = None
            continue
        images = sorted(class_dir.glob("*.jpg"))
        real_images = [p for p in images if not p.name.startswith("aug_")]
        candidate = real_images[0] if real_images else (images[0] if images else None)
        if candidate is None:
            result[cls] = None
            continue
        try:
            with Image.open(candidate) as img:
                thumb = img.convert("RGB").resize((300, 300))
                buf = io.BytesIO()
                thumb.save(buf, format="JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                result[cls] = f"data:image/jpeg;base64,{b64}"
        except Exception:
            result[cls] = None

    if any(v is not None for v in result.values()):
        try:
            with open(cached_json, "w", encoding="utf-8") as f:
                json.dump(result, f)
        except Exception:
            pass

    _class_images = result
    return result


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "efficientnetv2s_available": EFFIENTNET_CHECKPOINT.exists(),
        "vit_base_available": VIT_CHECKPOINT.exists(),
        "efficientnetv2s_loaded": "efficientnetv2s" in _models,
        "vit_base_loaded": "vit_base" in _models,
        "active_models": list(_models.keys()),
    }


@app.get("/api/classes")
def get_classes():
    """Return 6 classes with disease descriptions, symptoms, management, and representative images."""
    class_images = _get_class_images()

    descriptions = {
        "Bacterial Leaf Blight": {
            "desc": "A devastating bacterial disease caused by Xanthomonas oryzae pv. oryzae.",
            "symptoms": "Water-soaked to yellowish stripes along leaf margins progressing inward, turning gray-white with wavy edges. Droplets of bacterial ooze appear during humid mornings.",
            "management": "Use certified disease-free seed, maintain balanced nitrogen application, avoid excess standing water, and apply copper-based bactericides or plant resistance inducers."
        },
        "Brown Spot": {
            "desc": "A prevalent fungal pathogen caused by Cochliobolus miyabeanus (Bipolaris oryzae).",
            "symptoms": "Round to oval brown spots with yellow halos; severe attacks cause seedling blight and glume discoloration resulting in unfilled grains.",
            "management": "Improve soil fertility (adequate potassium and silicon), use hot-water seed treatment (53-54°C for 10 min), and apply propiconazole or iprodione fungicides when threshold is reached."
        },
        "Healthy Rice Leaf": {
            "desc": "Clean, uninfected foliage displaying vigorous uniform green coloration.",
            "symptoms": "Intact leaf blades, no necrotic spotting, water-soaking, or fungal sporulation. High photosynthetic potential.",
            "management": "Maintain recommended crop spacing, optimal water management, and scout regularly to preserve natural predator populations and avoid unnecessary chemical sprays."
        },
        "Leaf Blast": {
            "desc": "One of the most destructive rice fungal diseases worldwide, caused by Magnaporthe oryzae.",
            "symptoms": "Distinctive spindle-shaped or diamond-shaped lesions with grayish centers and dark brown margins. Lesions coalesce rapidly under high nitrogen and humidity.",
            "management": "Plant resistant cultivars, avoid excessive split nitrogen fertilizer, manage field water depths, and apply preventive tricyclazole, azoxystrobin, or isoprothiolane."
        },
        "Leaf Scald": {
            "desc": "Caused by the fungal pathogen Microdochium oryzae (Gerlachia oryzae).",
            "symptoms": "Zonate brown lesions beginning at leaf tips and margins with alternating light and dark brown bands (scald-like bands).",
            "management": "Crop rotation, seed dressing with carbendazim or thiram, destruction of stubble, and timely application of benomyl or propiconazole."
        },
        "Sheath Blight": {
            "desc": "Soil-borne fungal disease caused by Rhizoctonia solani AG-1 IA.",
            "symptoms": "Oval or irregular greenish-gray water-soaked spots on leaf sheaths near the water line, rapidly enlarging into snake-skin patterns moving up the canopy.",
            "management": "Avoid dense transplanting, drain excess water periodically to lower canopy humidity, and spray validamycin, azoxystrobin, or trifloxystrobin at early panicle initiation."
        },
    }

    return [
        {
            "name": cls,
            "index": i,
            "description": descriptions.get(cls, {}).get("desc", ""),
            "symptoms": descriptions.get(cls, {}).get("symptoms", ""),
            "management": descriptions.get(cls, {}).get("management", ""),
            "image": class_images.get(cls),
        }
        for i, cls in enumerate(CLASS_NAMES)
    ]


@app.get("/api/models")
def get_models():
    """Return model efficiency, parameter counts, and test accuracy metrics."""
    def _read_json(path: Path) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    eff_metrics = _read_json(OUTPUTS_DIR / "efficientnetv2s" / "metrics.json")
    vit_metrics = _read_json(OUTPUTS_DIR / "vit_base" / "metrics.json")

    eff_efficiency = _read_json(OUTPUTS_DIR / "efficientnetv2s" / "efficiency.json")
    vit_efficiency = _read_json(OUTPUTS_DIR / "vit_base" / "efficiency.json")

    # Fallback default values
    if not eff_metrics:
        eff_metrics = {
            "model": "efficientnetv2s",
            "test_sample_count": 570,
            "accuracy": 0.9526,
            "precision_macro": 0.9523,
            "recall_macro": 0.9522,
            "f1_score_macro": 0.9521,
            "roc_auc_macro": 0.9961,
        }
    if not eff_efficiency:
        eff_efficiency = {
            "model_architecture": "EfficientNetV2-S",
            "total_parameters": 20660838,
            "trainable_parameters": 20506966,
            "model_file_size_mb": 83.04,
            "training_time_minutes": 9.98,
            "inference_latency": {
                "profiled_samples": 50,
                "average_inference_time_ms": 14.88,
                "throughput_fps": 67.2
            }
        }

    if not vit_metrics:
        vit_metrics = {
            "model": "vit_base",
            "test_sample_count": 570,
            "accuracy": 0.9772,
            "precision_macro": 0.9780,
            "recall_macro": 0.9770,
            "f1_score_macro": 0.9771,
            "roc_auc_macro": 0.9974,
        }
    if not vit_efficiency:
        vit_efficiency = {
            "model_architecture": "ViT-B/16 (Vision Transformer)",
            "total_parameters": 85803270,
            "trainable_parameters": 85803270,
            "model_file_size_mb": 982.52,
            "training_time_minutes": 4.75,
            "inference_latency": {
                "profiled_samples": 50,
                "average_inference_time_ms": 7.64,
                "throughput_fps": 130.8
            }
        }

    def _normalize_efficiency(
        raw: dict,
        arch_name: str,
        default_size_mb: float,
        default_train_min: float,
        checkpoint_path: Path,
    ) -> dict:
        size_mb = raw.get("model_file_size_mb")
        if size_mb is None:
            size_mb = (
                round(checkpoint_path.stat().st_size / (1024 * 1024), 2)
                if checkpoint_path.exists()
                else default_size_mb
            )
        latency = raw.get("inference_latency") or {}
        avg_ms = latency.get("average_inference_time_ms", raw.get("average_latency_ms", 0.0))
        fps = latency.get("throughput_fps", raw.get("throughput_fps", 0.0))
        return {
            "model_architecture": raw.get("model_architecture", raw.get("model_name", arch_name)),
            "total_parameters": raw.get("total_parameters", 0),
            "trainable_parameters": raw.get("trainable_parameters", 0),
            "model_file_size_mb": size_mb,
            "training_time_minutes": raw.get("training_time_minutes", default_train_min),
            "inference_latency": {
                "profiled_samples": latency.get("profiled_samples", 50),
                "average_inference_time_ms": avg_ms,
                "throughput_fps": fps,
            },
        }

    eff_efficiency = _normalize_efficiency(
        eff_efficiency, "EfficientNetV2-S", 83.04, 9.98, EFFIENTNET_CHECKPOINT
    )
    vit_efficiency = _normalize_efficiency(
        vit_efficiency, "ViT-B/16 (Vision Transformer)", 982.52, 4.75, VIT_CHECKPOINT
    )

    return {
        "efficientnetv2s": {
            "available": EFFIENTNET_CHECKPOINT.exists(),
            "metrics": eff_metrics,
            "efficiency": eff_efficiency,
        },
        "vit_base": {
            "available": VIT_CHECKPOINT.exists(),
            "metrics": vit_metrics,
            "efficiency": vit_efficiency,
        },
    }


@app.get("/api/metrics")
def get_metrics():
    """Return per-class classification report and confusion matrix for both models."""
    import csv

    def _read_report(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("", row.get("Unnamed: 0", ""))
                if label in CLASS_NAMES:
                    rows.append({
                        "class": label,
                        "precision": float(row.get("precision", 0)),
                        "recall": float(row.get("recall", 0)),
                        "f1_score": float(row.get("f1-score", 0)),
                        "support": int(float(row.get("support", 0))),
                    })
        return rows

    def _read_confusion(path: Path) -> Dict[str, Dict[str, int]]:
        if not path.exists():
            return {}
        result: Dict[str, Dict[str, int]] = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            col_names = header[1:]
            for row in reader:
                if not row:
                    continue
                actual = row[0]
                result[actual] = {col_names[i]: int(row[i + 1]) for i in range(len(col_names))}
        return result

    eff_report_path = OUTPUTS_DIR / "efficientnetv2s" / "classification_report.csv"
    eff_cm_path = OUTPUTS_DIR / "efficientnetv2s" / "confusion_matrix.csv"
    if not eff_cm_path.exists():
        eff_cm_path = OUTPUTS_DIR / "efficientnetv2s" / "efficientnetv2s_confusion_matrix.csv"

    vit_report_path = OUTPUTS_DIR / "vit_base" / "classification_report.csv"
    vit_cm_path = OUTPUTS_DIR / "vit_base" / "confusion_matrix.csv"
    if not vit_cm_path.exists():
        vit_cm_path = OUTPUTS_DIR / "vit_base" / "vit_base_confusion_matrix.csv"

    return {
        "efficientnetv2s": {
            "per_class": _read_report(eff_report_path),
            "confusion_matrix": _read_confusion(eff_cm_path),
        },
        "vit_base": {
            "per_class": _read_report(vit_report_path),
            "confusion_matrix": _read_confusion(vit_cm_path),
        },
        "class_names": CLASS_NAMES,
    }


@app.get("/api/sample-results")
def get_sample_results():
    """Return pre-computed predictions (both models) for the 6 sample class images."""
    path = OUTPUTS_DIR / "sample_results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample results not generated yet. Run generate_sample_results.py locally.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query("both", description="Model to use: 'both' (default), 'efficientnetv2s', or 'vit_base'"),
):
    """
    Accepts an uploaded image and performs inference and XAI generation.
    Supports Dual Mode ('both') for one-click side-by-side comparison!
    """
    if model not in ("both", "efficientnetv2s", "vit_base"):
        raise HTTPException(status_code=400, detail="model must be 'both', 'efficientnetv2s', or 'vit_base'")

    # In single-model deployments, redirect "both" to the available model
    if SINGLE_MODEL:
        if model == "both" or model == SINGLE_MODEL:
            model = SINGLE_MODEL
        else:
            raise HTTPException(status_code=400, detail=f"Only '{SINGLE_MODEL}' is available in this deployment.")

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are accepted.")

    # Max 12 MB
    contents = await file.read()
    if len(contents) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be smaller than 12 MB.")

    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Please ensure valid image file.")

    disclaimer = (
        "Predictions, Grad-CAM, and ViT Saliency heatmaps are deep learning research outputs intended for decision-support. "
        "Heatmaps highlight key lesion features influencing classification. Results should be verified with field agronomists."
    )

    if model == "both":
        eff_res = _run_single_inference(img, "efficientnetv2s")
        vit_res = _run_single_inference(img, "vit_base")
        return {
            "mode": "both",
            "efficientnetv2s": eff_res,
            "vit_base": vit_res,
            "disclaimer": disclaimer,
        }
    else:
        single_res = _run_single_inference(img, model)
        return {
            "mode": "single",
            **single_res,
            "disclaimer": disclaimer,
        }


# ─── Single-Server SPA Static Files Mount ────────────────────────────────────

if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Don't intercept API paths
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)

        index_file = FRONTEND_DIST / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)

        raise HTTPException(status_code=404, detail="Page not found")
