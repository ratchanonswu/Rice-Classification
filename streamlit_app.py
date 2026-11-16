"""
RiceVision AI — Streamlit Deployment App (Styled to match RiceVision AI React UI)
Supports both EfficientNetV2-S (Grad-CAM++) and ViT-B/16 (14x14 Attention Saliency)
alongside benchmark comparison, disease reference, and methodology documentation.
"""
import base64
import csv
import gc
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import keras_hub  # noqa: F401 — required for ViT-B/16 custom layer deserialization
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

# ─── Page Config & Constants ──────────────────────────────────────────────────

st.set_page_config(
    page_title="RiceVision AI — EfficientNetV2-S vs. ViT-B/16",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

EFFNET_CHECKPOINT = MODELS_DIR / "efficientnetv2s.keras"
VIT_CHECKPOINT = MODELS_DIR / "vit_base.keras"

CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight",
]

CLASS_ABBR = {
    "Bacterial Leaf Blight": "BLB",
    "Brown Spot": "BS",
    "Healthy Rice Leaf": "HRL",
    "Leaf Blast": "LB",
    "Leaf Scald": "LS",
    "Sheath Blight": "SB",
}

DISEASE_INFO = {
    "Bacterial Leaf Blight": {
        "desc": "A devastating bacterial disease caused by Xanthomonas oryzae pv. oryzae.",
        "symptoms": "Water-soaked to yellowish stripes along leaf margins progressing inward, turning gray-white with wavy edges.",
        "management": "Use certified disease-free seed, maintain balanced nitrogen, avoid excess standing water, and apply copper-based bactericides.",
    },
    "Brown Spot": {
        "desc": "A prevalent fungal pathogen caused by Cochliobolus miyabeanus (Bipolaris oryzae).",
        "symptoms": "Round to oval brown spots with yellow halos; severe attacks cause seedling blight and unfilled grains.",
        "management": "Improve soil fertility (adequate potassium and silicon), use hot-water seed treatment (53-54°C for 10 min), and apply propiconazole if needed.",
    },
    "Healthy Rice Leaf": {
        "desc": "Clean, uninfected foliage displaying vigorous uniform green coloration.",
        "symptoms": "Intact leaf blades, no necrotic spotting, water-soaking, or fungal sporulation.",
        "management": "Maintain recommended crop spacing, optimal water management, and scout regularly.",
    },
    "Leaf Blast": {
        "desc": "One of the most destructive rice fungal diseases worldwide, caused by Magnaporthe oryzae.",
        "symptoms": "Spindle-shaped or diamond-shaped lesions with grayish centers and dark brown margins.",
        "management": "Plant resistant cultivars, avoid excessive split nitrogen fertilizer, and apply preventive tricyclazole or azoxystrobin.",
    },
    "Leaf Scald": {
        "desc": "Caused by the fungal pathogen Microdochium oryzae (Gerlachia oryzae).",
        "symptoms": "Zonate brown lesions beginning at leaf tips and margins with alternating light and dark brown bands.",
        "management": "Crop rotation, seed dressing with carbendazim or thiram, destruction of stubble, and timely fungicide application.",
    },
    "Sheath Blight": {
        "desc": "Soil-borne fungal disease caused by Rhizoctonia solani AG-1 IA.",
        "symptoms": "Oval or irregular greenish-gray water-soaked spots on leaf sheaths enlarging into snake-skin patterns.",
        "management": "Avoid dense transplanting, drain excess water periodically to lower canopy humidity, and spray validamycin or azoxystrobin.",
    },
}

DISCLAIMER = (
    "Predictions, Grad-CAM++, and ViT Attention Saliency heatmaps are deep learning research outputs "
    "intended for decision-support and should be verified with qualified agronomists before field treatment."
)

# ─── Custom CSS (Matching RiceVision AI React + Tailwind UI) ──────────────────

st.markdown(
    """
    <style>
    /* Clean background & layout */
    .stApp {
        background-color: #f8fafc;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }
    header[data-testid="stHeader"] {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid #e2e8f0;
    }
    #MainMenu, footer {visibility: hidden;}

    /* Hero Banner */
    .rv-hero {
        background: linear-gradient(135deg, #14532d 0%, #15803d 55%, #166534 100%);
        color: white;
        padding: 2rem 2.25rem;
        border-radius: 1.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(21, 128, 61, 0.22);
    }
    .rv-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.25);
        color: #dcfce7;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        margin-bottom: 0.75rem;
    }
    .rv-hero h1 {
        color: white !important;
        font-size: 1.95rem !important;
        font-weight: 800 !important;
        margin: 0 0 0.5rem 0 !important;
        line-height: 1.2 !important;
    }
    .rv-hero p {
        color: #dcfce7;
        font-size: 0.96rem;
        margin: 0;
        max-width: 780px;
        line-height: 1.55;
    }

    /* Stat Cards */
    .rv-stat {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .rv-stat-accent {
        background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%);
        border: 1px solid #bbf7d0;
    }
    .rv-stat-purple {
        background: linear-gradient(180deg, #faf5ff 0%, #ffffff 100%);
        border: 1px solid #e9d5ff;
    }
    .rv-stat-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .rv-stat-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.2rem 0;
    }
    .rv-stat-sub {
        font-size: 0.78rem;
        color: #475569;
    }

    /* Model Summary Cards */
    .rv-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }
    .rv-card-green {
        background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 45%);
        border: 1px solid #bbf7d0;
    }
    .rv-card-purple {
        background: linear-gradient(180deg, #faf5ff 0%, #ffffff 45%);
        border: 1px solid #e9d5ff;
    }
    .rv-pill-green {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 9999px;
    }
    .rv-pill-purple {
        display: inline-block;
        background: #f3e8ff;
        color: #6b21a8;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 9999px;
    }

    /* Top Nav Radio as clean segmented pills */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        background: #ffffff;
        padding: 0.35rem;
        border-radius: 0.9rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        justify-content: center;
        gap: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Cached Data Loaders ──────────────────────────────────────────────────────

@st.cache_data
def load_json(path_str: str) -> Any:
    path = Path(path_str)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_classification_report(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("", row.get("Unnamed: 0", ""))
            if label in CLASS_NAMES:
                rows.append({
                    "Class": label,
                    "Abbr": CLASS_ABBR[label],
                    "Precision": float(row.get("precision", 0)) * 100,
                    "Recall": float(row.get("recall", 0)) * 100,
                    "F1-Score": float(row.get("f1-score", 0)) * 100,
                    "Support": int(float(row.get("support", 0))),
                })
    return pd.DataFrame(rows)


@st.cache_data
def load_confusion_matrix(model_dir: str) -> pd.DataFrame:
    cm_path = OUTPUTS_DIR / model_dir / "confusion_matrix.csv"
    if not cm_path.exists():
        return pd.DataFrame()
    return pd.read_csv(cm_path, index_col=0)


# ─── Model Loading & XAI ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_cached(model_key: str) -> Any:
    ckpt = EFFNET_CHECKPOINT if model_key == "efficientnetv2s" else VIT_CHECKPOINT
    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")
    return tf.keras.models.load_model(str(ckpt), compile=False)


def compute_gradcam_plusplus(img_tensor: tf.Tensor, model: Any, pred_index: Optional[int] = None) -> np.ndarray:
    """Chattopadhay et al. (2018) Grad-CAM++ using pre-softmax logits for EfficientNetV2-S."""
    x_in = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
    backbone = model.layers[1]
    head_layers = model.layers[2:]

    with tf.GradientTape() as tape:
        features = backbone(x_in, training=False)
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


def compute_vit_saliency(img_tensor: tf.Tensor, vit_model: Any) -> np.ndarray:
    """14x14 patch-pooled input-gradient attention saliency for ViT-B/16."""
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
    p99 = np.percentile(grid, 99)
    grid = np.clip(grid, 0, p99 if p99 > 0 else grid.max())
    if grid.max() > 0:
        grid = grid / grid.max()
    return grid


def make_heatmap_images(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Tuple[Image.Image, Image.Image]:
    orig = np.array(pil_img.convert("RGB"))
    h, w, _ = orig.shape
    hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_color = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(orig * (1 - alpha) + hm_color * alpha)
    return Image.fromarray(overlay), Image.fromarray(hm_color)


def run_inference(pil_img: Image.Image, model_key: str, alpha: float) -> Dict[str, Any]:
    model = load_model_cached(model_key)
    resized = pil_img.convert("RGB").resize((224, 224))
    arr = np.array(resized, dtype=np.float32)

    if model_key == "vit_base":
        arr = (arr / 127.5) - 1.0
    tensor = np.expand_dims(arr, axis=0)

    t0 = time.perf_counter()
    preds = model.predict(tensor, verbose=0)
    probs = preds[0]
    if float(np.sum(probs)) > 1.05 or float(np.min(probs)) < 0.0:
        probs = tf.nn.softmax(probs).numpy()
    inference_ms = round((time.perf_counter() - t0) * 1000, 2)

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    scores = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}

    if model_key == "efficientnetv2s":
        heatmap = compute_gradcam_plusplus(tf.convert_to_tensor(tensor), model, pred_idx)
        xai_method = "Grad-CAM++"
    else:
        heatmap = compute_vit_saliency(tf.convert_to_tensor(tensor), model)
        xai_method = "Attention Saliency (14×14 Patches)"

    overlay_img, raw_hm_img = make_heatmap_images(resized, heatmap, alpha=alpha)
    return {
        "model": model_key,
        "predicted_class": pred_class,
        "confidence": confidence,
        "scores": scores,
        "inference_ms": inference_ms,
        "xai_method": xai_method,
        "overlay_img": overlay_img,
        "raw_hm_img": raw_hm_img,
        "orig_img": resized,
    }


def b64_to_pil(data_url: str) -> Image.Image:
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def render_stat_card(label: str, value: str, sub: str, variant: str = "default") -> str:
    cls = "rv-stat"
    if variant == "green":
        cls += " rv-stat-accent"
    elif variant == "purple":
        cls += " rv-stat-purple"
    return (
        f'<div class="{cls}">'
        f'<div class="rv-stat-label">{label}</div>'
        f'<div class="rv-stat-value">{value}</div>'
        f'<div class="rv-stat-sub">{sub}</div>'
        f'</div>'
    )


# ─── Sidebar (System / RAM Utilities) ─────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Runtime Status")
    st.write(f"• EfficientNetV2-S (`83.0 MB`): {'✅ Ready' if EFFNET_CHECKPOINT.exists() else '❌ Missing'}")
    st.write(f"• ViT-B/16 (`982.5 MB`): {'✅ Ready' if VIT_CHECKPOINT.exists() else '❌ Missing'}")
    if st.button("🧹 Free Model RAM Cache", use_container_width=True):
        load_model_cached.clear()
        tf.keras.backend.clear_session()
        gc.collect()
        st.success("Cleared cached models from RAM.")


# ─── Top Navigation Bar (Matching React Navbar) ───────────────────────────────

nav_col1, nav_col2 = st.columns([1.2, 3.8])
with nav_col1:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.5rem;padding-top:0.35rem;">'
        '<span style="font-size:1.5rem;">🌾</span>'
        '<div><div style="font-weight:800;font-size:1.05rem;color:#14532d;line-height:1.1;">RiceVision AI</div>'
        '<div style="font-size:0.72rem;color:#64748b;">EfficientNetV2-S vs. ViT-B/16</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
with nav_col2:
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "📊 Comparison", "🌱 Classes", "🔬 Predict & XAI", "🧪 Methodology"],
        index=3,
        horizontal=True,
        label_visibility="collapsed",
    )

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ─── Pages ────────────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    st.markdown(
        """
        <div class="rv-hero">
            <div class="rv-badge">Zero-Leakage SHA-256 + pHash Benchmark • Explainable AI</div>
            <h1>Rice Leaf Disease Classification:<br/>EfficientNetV2-S vs. Vision Transformer (ViT-B/16)</h1>
            <p>
                Controlled empirical benchmark comparing a lightweight convolutional backbone against a global
                self-attention Vision Transformer across 3,798 deduplicated leaf specimens and 6 disease classes,
                instrumented with Grad-CAM++ and 14×14 Attention Saliency.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(render_stat_card("ViT-B/16 Test Acc", "97.72%", "Transformer Leader (+2.46%)", "purple"), unsafe_allow_html=True)
    c2.markdown(render_stat_card("EfficientNetV2-S Acc", "95.26%", "Lightweight CNN Baseline", "green"), unsafe_allow_html=True)
    c3.markdown(render_stat_card("Model Size Gap", "11.8× Lighter", "83.04 MB vs 982.52 MB", "green"), unsafe_allow_html=True)
    c4.markdown(render_stat_card("ViT GPU Throughput", "130.8 FPS", "7.64 ms / image on GPU", "purple"), unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class="rv-card rv-card-green">
                <span class="rv-pill-green">Lightweight CNN</span>
                <h3 style="margin:0.5rem 0 0.4rem 0;color:#14532d;">EfficientNetV2-S</h3>
                <p style="font-size:0.88rem;color:#475569;margin-bottom:0.75rem;">
                    Fused-MBConv convolutional backbone optimized for compact deployment and low RAM footprints.
                </p>
                <ul style="font-size:0.85rem;color:#334155;margin:0;padding-left:1.2rem;">
                    <li><strong>Test Accuracy:</strong> 95.26% &nbsp;|&nbsp; <strong>Macro F1:</strong> 95.21% &nbsp;|&nbsp; <strong>ROC-AUC:</strong> 99.61%</li>
                    <li><strong>Checkpoint:</strong> 83.04 MB (20,660,838 parameters)</li>
                    <li><strong>XAI Attribution:</strong> Pre-softmax Grad-CAM++ (Chattopadhay et al., 2018)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="rv-card rv-card-purple">
                <span class="rv-pill-purple">Vision Transformer</span>
                <h3 style="margin:0.5rem 0 0.4rem 0;color:#581c87;">ViT-B/16 (Patch 16×16)</h3>
                <p style="font-size:0.88rem;color:#475569;margin-bottom:0.75rem;">
                    12-layer Multi-Head Self-Attention encoder capturing global leaf context to eliminate background shortcut bias.
                </p>
                <ul style="font-size:0.85rem;color:#334155;margin:0;padding-left:1.2rem;">
                    <li><strong>Test Accuracy:</strong> 97.72% &nbsp;|&nbsp; <strong>Macro F1:</strong> 97.71% &nbsp;|&nbsp; <strong>ROC-AUC:</strong> 99.74%</li>
                    <li><strong>Checkpoint:</strong> 982.52 MB (85,803,270 parameters)</li>
                    <li><strong>XAI Attribution:</strong> 14×14 Patch Input-Gradient Attention Saliency</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


elif page == "🔬 Predict & XAI":
    st.markdown("### 🔬 Dual-Model Disease Predictor & Explainable AI")
    st.caption("Upload a rice leaf specimen or inspect pre-computed 6-class test samples with Grad-CAM++ and ViT Attention Saliency.")

    tab_live, tab_sample = st.tabs(["📤 Upload Image (Live Inference)", "🖼️ Pre-Computed 6-Class Gallery"])

    with tab_live:
        with st.container(border=True):
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1.4, 1.4])
            with col_ctrl1:
                uploaded = st.file_uploader("Upload Rice Leaf Photo (.jpg, .png, .webp)", type=["jpg", "jpeg", "png", "webp"])
            with col_ctrl2:
                model_choice = st.selectbox(
                    "Inference Mode",
                    ["Both Models (Side-by-Side)", "EfficientNetV2-S Only (Low RAM)", "ViT-B/16 Only"],
                )
            with col_ctrl3:
                alpha = st.slider("Heatmap Opacity (α)", 0.15, 0.85, 0.45, 0.05)
                view_mode = st.radio("Heatmap View", ["Overlay", "Raw Heatmap", "Original"], horizontal=True)

        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            if st.button("🚀 Run Diagnosis", type="primary", use_container_width=True):
                targets = (
                    ["efficientnetv2s", "vit_base"]
                    if model_choice.startswith("Both")
                    else (["efficientnetv2s"] if model_choice.startswith("EfficientNet") else ["vit_base"])
                )
                results = {}
                for mk in targets:
                    label = "EfficientNetV2-S" if mk == "efficientnetv2s" else "ViT-B/16"
                    with st.spinner(f"Running {label}..."):
                        results[mk] = run_inference(pil_img, mk, alpha)

                if len(results) == 2:
                    e_cls = results["efficientnetv2s"]["predicted_class"]
                    v_cls = results["vit_base"]["predicted_class"]
                    if e_cls == v_cls:
                        st.success(f"🤝 **Model Consensus:** Both models agree on **{v_cls}**")
                    else:
                        st.warning(f"⚖️ **Divergence:** EfficientNetV2-S predicts **{e_cls}**, while ViT-B/16 predicts **{v_cls}**")

                cols = st.columns(len(results))
                for col, (mk, res) in zip(cols, results.items()):
                    title = "EfficientNetV2-S (Grad-CAM++)" if mk == "efficientnetv2s" else "ViT-B/16 (Attention Saliency)"
                    with col:
                        with st.container(border=True):
                            st.subheader(title)
                            img_to_show = (
                                res["overlay_img"] if view_mode == "Overlay"
                                else (res["raw_hm_img"] if view_mode == "Raw Heatmap" else res["orig_img"])
                            )
                            st.image(img_to_show, use_container_width=True, caption=f"{res['xai_method']} • {res['inference_ms']} ms")
                            st.metric("Predicted Class", res["predicted_class"], f"{res['confidence']*100:.2f}% confidence")
                            score_df = pd.DataFrame({
                                "Class": list(res["scores"].keys()),
                                "Probability (%)": [v * 100 for v in res["scores"].values()],
                            }).sort_values("Probability (%)", ascending=True)
                            fig = px.bar(score_df, x="Probability (%)", y="Class", orientation="h", height=230)
                            fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), xaxis_range=[0, 100])
                            st.plotly_chart(fig, use_container_width=True)
                st.info(DISCLAIMER)

    with tab_sample:
        samples = load_json(str(OUTPUTS_DIR / "sample_results.json"))
        class_imgs = load_json(str(OUTPUTS_DIR / "class_images.json"))
        if not samples:
            st.warning("`outputs/sample_results.json` not found.")
        else:
            scol1, scol2 = st.columns([2, 1])
            with scol1:
                chosen = st.selectbox("Select Pre-Computed Test Sample", list(samples.keys()))
            with scol2:
                sample_hm_view = st.radio("Gallery Heatmap View", ["Overlay", "Raw Heatmap"], horizontal=True)

            item = samples[chosen]
            eff = item["efficientnetv2s"]
            vit = item["vit_base"]
            hm_key = "gradcam_overlay" if sample_hm_view == "Overlay" else "gradcam_raw"

            if eff["predicted_class"] == vit["predicted_class"]:
                st.success(f"🤝 **Model Consensus:** Both models predict **{vit['predicted_class']}** (True Class: **{chosen}**)")
            else:
                st.warning(f"⚖️ **Divergence:** EfficientNetV2-S predicts **{eff['predicted_class']}**, ViT-B/16 predicts **{vit['predicted_class']}** (True Class: **{chosen}**)")

            c1, c2, c3 = st.columns(3)
            with c1:
                with st.container(border=True):
                    st.subheader("Original Leaf")
                    orig_b64 = item.get("original_image") or class_imgs.get(chosen)
                    if orig_b64:
                        st.image(b64_to_pil(orig_b64), caption=f"True Class: {chosen}", use_container_width=True)
                    else:
                        st.info("Original thumbnail unavailable.")

            for col, (res, title, xai_label, bar_color) in zip([c2, c3], [
                (eff, "EfficientNetV2-S", "Grad-CAM++", "#16a34a"),
                (vit, "ViT-B/16 (Transformer)", "Attention Saliency (14×14)", "#9333ea"),
            ]):
                with col:
                    with st.container(border=True):
                        st.subheader(title)
                        st.image(
                            b64_to_pil(res[hm_key]),
                            caption=f"{xai_label} • {res['inference_ms']} ms",
                            use_container_width=True,
                        )
                        st.metric("Predicted Class", res["predicted_class"], f"{res['confidence']*100:.2f}% confidence")
                        score_df = pd.DataFrame({
                            "Class": list(res["scores"].keys()),
                            "Probability (%)": [v * 100 for v in res["scores"].values()],
                        }).sort_values("Probability (%)", ascending=True)
                        fig = px.bar(score_df, x="Probability (%)", y="Class", orientation="h", height=220, color_discrete_sequence=[bar_color])
                        fig.update_layout(margin=dict(l=0, r=10, t=10, b=0), xaxis_range=[0, 100])
                        st.plotly_chart(fig, use_container_width=True)


elif page == "📊 Comparison":
    st.markdown("### 📊 Head-to-Head Model Benchmark")
    st.caption("Empirical comparison between EfficientNetV2-S (Lightweight CNN) and ViT-B/16 (Vision Transformer) on 570 held-out test images.")

    eff_m = load_json(str(OUTPUTS_DIR / "efficientnetv2s" / "metrics.json"))
    vit_m = load_json(str(OUTPUTS_DIR / "vit_base" / "metrics.json"))
    eff_e = load_json(str(OUTPUTS_DIR / "efficientnetv2s" / "efficiency.json"))
    vit_e = load_json(str(OUTPUTS_DIR / "vit_base" / "efficiency.json"))

    tab1, tab2, tab3 = st.tabs(["Benchmark Overview", "Per-Class Metrics", "Confusion Matrices"])
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(render_stat_card("ViT-B/16 Test Acc", f"{vit_m.get('accuracy', 0.9772)*100:.2f}%", "Transformer Leader (+2.46%)", "purple"), unsafe_allow_html=True)
        c2.markdown(render_stat_card("EfficientNetV2-S Acc", f"{eff_m.get('accuracy', 0.9526)*100:.2f}%", "Lightweight CNN Baseline", "green"), unsafe_allow_html=True)
        c3.markdown(render_stat_card("Model Size Gap", "11.8× Lighter", "83.04 MB vs 982.52 MB", "green"), unsafe_allow_html=True)
        c4.markdown(render_stat_card("ViT Throughput", f"{vit_e.get('throughput_fps', 130.8)} FPS", "Sub-8ms Latency on GPU", "purple"), unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        left_col, right_col = st.columns(2)
        with left_col:
            with st.container(border=True):
                st.subheader("Classification & Efficiency Metrics")
                comp_df = pd.DataFrame([
                    {"Metric": "Test Accuracy", "EfficientNetV2-S": f"{eff_m.get('accuracy', 0.9526)*100:.2f}%", "ViT-B/16": f"{vit_m.get('accuracy', 0.9772)*100:.2f}%", "Winner": "ViT-B/16"},
                    {"Metric": "Macro Precision", "EfficientNetV2-S": f"{eff_m.get('precision_macro', 0.9523)*100:.2f}%", "ViT-B/16": f"{vit_m.get('precision_macro', 0.9780)*100:.2f}%", "Winner": "ViT-B/16"},
                    {"Metric": "Macro Recall", "EfficientNetV2-S": f"{eff_m.get('recall_macro', 0.9522)*100:.2f}%", "ViT-B/16": f"{vit_m.get('recall_macro', 0.9770)*100:.2f}%", "Winner": "ViT-B/16"},
                    {"Metric": "Macro F1-Score", "EfficientNetV2-S": f"{eff_m.get('f1_score_macro', 0.9521)*100:.2f}%", "ViT-B/16": f"{vit_m.get('f1_score_macro', 0.9771)*100:.2f}%", "Winner": "ViT-B/16"},
                    {"Metric": "ROC-AUC (Macro)", "EfficientNetV2-S": f"{eff_m.get('roc_auc_macro', 0.9961)*100:.2f}%", "ViT-B/16": f"{vit_m.get('roc_auc_macro', 0.9974)*100:.2f}%", "Winner": "ViT-B/16"},
                    {"Metric": "Model Size (MB)", "EfficientNetV2-S": "83.04 MB", "ViT-B/16": "982.52 MB", "Winner": "EfficientNetV2-S"},
                    {"Metric": "Total Parameters", "EfficientNetV2-S": f"{eff_e.get('total_parameters', 20660838):,}", "ViT-B/16": f"{vit_e.get('total_parameters', 85803270):,}", "Winner": "EfficientNetV2-S"},
                    {"Metric": "Avg GPU Latency", "EfficientNetV2-S": f"{eff_e.get('average_latency_ms', 14.88)} ms", "ViT-B/16": f"{vit_e.get('average_latency_ms', 7.64)} ms", "Winner": "ViT-B/16"},
                    {"Metric": "GPU Throughput", "EfficientNetV2-S": f"{eff_e.get('throughput_fps', 67.2)} fps", "ViT-B/16": f"{vit_e.get('throughput_fps', 130.8)} fps", "Winner": "ViT-B/16"},
                ])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

        with right_col:
            with st.container(border=True):
                st.subheader("Performance Profile Radar")
                radar_cats = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
                eff_vals = [
                    eff_m.get("accuracy", 0.9526) * 100,
                    eff_m.get("precision_macro", 0.9523) * 100,
                    eff_m.get("recall_macro", 0.9522) * 100,
                    eff_m.get("f1_score_macro", 0.9521) * 100,
                    eff_m.get("roc_auc_macro", 0.9961) * 100,
                ]
                vit_vals = [
                    vit_m.get("accuracy", 0.9772) * 100,
                    vit_m.get("precision_macro", 0.9780) * 100,
                    vit_m.get("recall_macro", 0.9770) * 100,
                    vit_m.get("f1_score_macro", 0.9771) * 100,
                    vit_m.get("roc_auc_macro", 0.9974) * 100,
                ]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=eff_vals + [eff_vals[0]], theta=radar_cats + [radar_cats[0]], fill="toself", name="EfficientNetV2-S", line_color="#16a34a"))
                fig_radar.add_trace(go.Scatterpolar(r=vit_vals + [vit_vals[0]], theta=radar_cats + [radar_cats[0]], fill="toself", name="ViT-B/16", line_color="#9333ea"))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[90, 100])), height=370, margin=dict(l=30, r=30, t=20, b=20))
                st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        df_eff = load_classification_report(str(OUTPUTS_DIR / "efficientnetv2s" / "classification_report.csv"))
        df_vit = load_classification_report(str(OUTPUTS_DIR / "vit_base" / "classification_report.csv"))
        if not df_eff.empty and not df_vit.empty:
            merged = pd.DataFrame({
                "Class": df_eff["Class"],
                "EfficientNetV2-S F1 (%)": df_eff["F1-Score"].round(2),
                "ViT-B/16 F1 (%)": df_vit["F1-Score"].round(2),
                "Support": df_eff["Support"],
            })
            fig = go.Figure(data=[
                go.Bar(name="EfficientNetV2-S", x=merged["Class"], y=merged["EfficientNetV2-S F1 (%)"], marker_color="#16a34a"),
                go.Bar(name="ViT-B/16", x=merged["Class"], y=merged["ViT-B/16 F1 (%)"], marker_color="#9333ea"),
            ])
            fig.update_layout(barmode="group", yaxis_range=[85, 100], height=400, title="F1-Score Comparison Across All 6 Disease Classes")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(merged, use_container_width=True, hide_index=True)

    with tab3:
        c1, c2 = st.columns(2)
        for col, (mdir, title, scale) in zip([c1, c2], [
            ("efficientnetv2s", "EfficientNetV2-S (570 Test Images)", "Greens"),
            ("vit_base", "ViT-B/16 (570 Test Images)", "Purples"),
        ]):
            cm = load_confusion_matrix(mdir)
            if not cm.empty:
                with col:
                    with st.container(border=True):
                        fig = px.imshow(cm, text_auto=True, color_continuous_scale=scale, title=title)
                        fig.update_layout(height=420)
                        st.plotly_chart(fig, use_container_width=True)


elif page == "🌱 Classes":
    st.markdown("### 🌱 Rice Leaf Pathology Reference (6 Classes)")
    st.caption("Representative specimens, visual symptoms, and agronomic management guidelines.")
    class_imgs = load_json(str(OUTPUTS_DIR / "class_images.json"))
    cols = st.columns(2)
    for i, cls in enumerate(CLASS_NAMES):
        info = DISEASE_INFO[cls]
        with cols[i % 2]:
            with st.container(border=True):
                c_img, c_txt = st.columns([1, 2])
                with c_img:
                    if cls in class_imgs and class_imgs[cls]:
                        st.image(b64_to_pil(class_imgs[cls]), use_container_width=True)
                with c_txt:
                    st.subheader(cls)
                    st.write(info["desc"])
                    st.markdown(f"**Symptoms:** {info['symptoms']}")
                    st.markdown(f"**Management:** {info['management']}")


elif page == "🧪 Methodology":
    st.markdown("### 🧪 Research Pipeline & Methodology")
    st.caption("End-to-end leakage-free workflow from raw image collection to Explainable AI deployment.")

    steps = [
        ("1. Dataset Collection & Standardization", "3,829 raw rice leaf images collected across 6 pathological classes."),
        ("2. Audit & Leakage-Free Splitting (SHA-256 + pHash)",
         "Excluded **31 exact duplicates** via SHA-256 (3,798 unique canonical images preserved) and grouped **67 near-duplicate clusters** via pHash (`d ≤ 10`) into the same split: **75% Train (2,847)** / **10% Val (381)** / **15% Test (570)** with **0 hash overlap**."),
        ("3. Two-Stage Fine-Tuning",
         "**Stage 1:** Freeze ImageNet-pretrained backbone, train classification head (`Adam`, `lr=1e-3`).\n\n"
         "**Stage 2:** Unfreeze full model with low learning rate (`AdamW`, `5e-5` for EfficientNetV2-S, `2e-5` for ViT-B/16), `ReduceLROnPlateau`, and `EarlyStopping`."),
        ("4. Explainable AI (Grad-CAM++ & Attention Saliency)",
         "**EfficientNetV2-S:** Pre-softmax **Grad-CAM++** (Chattopadhay et al., 2018).\n\n"
         "**ViT-B/16:** **14×14 Patch Attention Saliency** (`avg_pool2d` `16×16`, 99th percentile clipping) blended at `α = 0.45`."),
    ]
    for title, body in steps:
        with st.container(border=True):
            st.subheader(title)
            st.markdown(body)
