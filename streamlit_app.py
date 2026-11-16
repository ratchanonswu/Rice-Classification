"""
RiceVision AI — Streamlit Deployment App
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
from typing import Any, Dict, List, Optional, Tuple

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
    initial_sidebar_state="expanded",
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


# ─── Sidebar Navigation ───────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌾 RiceVision AI")
    st.caption("EfficientNetV2-S vs. ViT-B/16")
    page = st.radio(
        "Navigation",
        ["Overview", "Live Prediction & XAI", "Model Comparison", "Disease Classes", "Methodology"],
        index=1,
    )
    st.divider()
    st.markdown("**Model Checkpoints**")
    st.write(f"• EfficientNetV2-S (`83.0 MB`): {'✅ Ready' if EFFNET_CHECKPOINT.exists() else '❌ Missing'}")
    st.write(f"• ViT-B/16 (`982.5 MB`): {'✅ Ready' if VIT_CHECKPOINT.exists() else '❌ Missing'}")
    if st.button("🧹 Free Model RAM Cache", use_container_width=True):
        load_model_cached.clear()
        tf.keras.backend.clear_session()
        gc.collect()
        st.success("Cleared cached models from RAM.")


# ─── Pages ────────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("🌾 Rice Leaf Disease Classification: EfficientNetV2-S vs. ViT-B/16")
    st.markdown(
        "Empirical benchmark comparing a lightweight CNN (**EfficientNetV2-S**) against a global "
        "self-attention **Vision Transformer (ViT-B/16)** on a SHA-256 + pHash deduplicated dataset."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ViT-B/16 Test Accuracy", "97.72%", "+2.46% vs CNN")
    c2.metric("EfficientNetV2-S Accuracy", "95.26%", "11.8× Lighter (83 MB)")
    c3.metric("ViT GPU Throughput", "130.8 FPS", "7.64 ms / image")
    c4.metric("Clean Dataset", "3,798 images", "6 Classes • 0 Leakage")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🟢 EfficientNetV2-S (Lightweight CNN)")
        st.markdown(
            "- **Test Accuracy:** `95.26%` | **Macro F1:** `95.21%` | **ROC-AUC:** `0.9961`\n"
            "- **Size & Params:** `83.04 MB` (`20,660,838` params)\n"
            "- **XAI Method:** Grad-CAM++ (Chattopadhay et al., 2018)\n"
            "- **Best For:** Resource-constrained & mobile/edge deployment"
        )
    with col_b:
        st.subheader("🟣 ViT-B/16 (Vision Transformer)")
        st.markdown(
            "- **Test Accuracy:** `97.72%` | **Macro F1:** `97.71%` | **ROC-AUC:** `0.9974`\n"
            "- **Size & Params:** `982.52 MB` (`85,803,270` params)\n"
            "- **XAI Method:** 14×14 Patch Attention Saliency\n"
            "- **Best For:** Maximum accuracy & eliminating background shortcut bias"
        )


elif page == "Live Prediction & XAI":
    st.title("🔬 Live Prediction & Explainable AI (XAI)")
    st.caption("Upload a rice leaf image or test with pre-computed clean test samples.")

    tab_live, tab_sample = st.tabs(["📤 Upload Image (Live Inference)", "🖼️ Pre-Computed 6-Class Gallery"])

    with tab_live:
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1.5, 1.5])
        with col_ctrl1:
            uploaded = st.file_uploader("Upload Rice Leaf Photo (.jpg, .png)", type=["jpg", "jpeg", "png", "webp"])
        with col_ctrl2:
            model_choice = st.selectbox(
                "Model Mode",
                ["Both Models (Side-by-Side)", "EfficientNetV2-S Only (Low RAM)", "ViT-B/16 Only"],
            )
        with col_ctrl3:
            alpha = st.slider("Heatmap Opacity (α)", 0.15, 0.85, 0.45, 0.05)
            view_mode = st.radio("Heatmap View", ["Overlay", "Raw Heatmap", "Original"], horizontal=True)

        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            if st.button("🚀 Run Diagnosis", type="primary"):
                targets = (
                    ["efficientnetv2s", "vit_base"]
                    if model_choice.startswith("Both")
                    else (["efficientnetv2s"] if model_choice.startswith("EfficientNet") else ["vit_base"])
                )
                results = {}
                for mk in targets:
                    label = "EfficientNetV2-S" if mk == "efficientnetv2s" else "ViT-B/16"
                    with st.spinner(f"Loading & running {label}..."):
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
        if not samples:
            st.warning("`outputs/sample_results.json` not found.")
        else:
            chosen = st.selectbox("Select Pre-Computed Test Sample", list(samples.keys()))
            item = samples[chosen]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(b64_to_pil(item["original_image"]), caption=f"Original ({chosen})", use_container_width=True)
            with c2:
                eff = item["efficientnetv2s"]
                st.image(b64_to_pil(eff["gradcam_overlay"]), caption=f"EfficientNetV2-S: {eff['predicted_class']} ({eff['confidence']*100:.2f}%)", use_container_width=True)
            with c3:
                vit = item["vit_base"]
                st.image(b64_to_pil(vit["gradcam_overlay"]), caption=f"ViT-B/16: {vit['predicted_class']} ({vit['confidence']*100:.2f}%)", use_container_width=True)


elif page == "Model Comparison":
    st.title("📊 Head-to-Head Model Benchmark")
    eff_m = load_json(str(OUTPUTS_DIR / "efficientnetv2s" / "metrics.json"))
    vit_m = load_json(str(OUTPUTS_DIR / "vit_base" / "metrics.json"))
    eff_e = load_json(str(OUTPUTS_DIR / "efficientnetv2s" / "efficiency.json"))
    vit_e = load_json(str(OUTPUTS_DIR / "vit_base" / "efficiency.json"))

    tab1, tab2, tab3 = st.tabs(["Benchmark Overview", "Per-Class Metrics", "Confusion Matrices"])
    with tab1:
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

    with tab2:
        df_eff = load_classification_report(str(OUTPUTS_DIR / "efficientnetv2s" / "classification_report.csv"))
        df_vit = load_classification_report(str(OUTPUTS_DIR / "vit_base" / "classification_report.csv"))
        if not df_eff.empty and not df_vit.empty:
            merged = pd.DataFrame({
                "Class": df_eff["Class"],
                "EfficientNetV2-S F1 (%)": df_eff["F1-Score"].round(2),
                "ViT-B/16 F1 (%)": df_vit["F1-Score"].round(2),
            })
            fig = go.Figure(data=[
                go.Bar(name="EfficientNetV2-S", x=merged["Class"], y=merged["EfficientNetV2-S F1 (%)"], marker_color="#16a34a"),
                go.Bar(name="ViT-B/16", x=merged["Class"], y=merged["ViT-B/16 F1 (%)"], marker_color="#9333ea"),
            ])
            fig.update_layout(barmode="group", yaxis_range=[85, 100], height=420)
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
                    fig = px.imshow(cm, text_auto=True, color_continuous_scale=scale, title=title)
                    fig.update_layout(height=420)
                    st.plotly_chart(fig, use_container_width=True)


elif page == "Disease Classes":
    st.title("🌱 Rice Leaf Pathology Reference (6 Classes)")
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


elif page == "Methodology":
    st.title("🧪 Research Pipeline & Methodology")
    st.markdown(
        """
        1. **Dataset Collection:** 3,829 raw rice leaf images across 6 classes.
        2. **Two-Stage Deduplication (`SHA-256` + `pHash`):**
           - Excluded **31 exact duplicates** via SHA-256 → **3,798 unique canonical images**.
           - Grouped **67 near-duplicate clusters** via pHash (`d <= 10`) into the same split.
           - Stratified split: **75% Train (2,847)** / **10% Val (381)** / **15% Test (570)** with **0 leakage**.
        3. **Two-Stage Fine-Tuning:**
           - **Stage 1:** Freeze pretrained backbone, train classification head (`Adam`, `lr=1e-3`).
           - **Stage 2:** Unfreeze full model with low LR (`AdamW`, `5e-5` for EfficientNetV2-S, `2e-5` for ViT-B/16), `ReduceLROnPlateau`, and `EarlyStopping`.
        4. **Explainable AI (XAI):**
           - **EfficientNetV2-S:** Pre-softmax **Grad-CAM++** (Chattopadhay et al., 2018).
           - **ViT-B/16:** **14×14 Patch Attention Saliency** (`avg_pool2d` `16×16`, 99th percentile clipping).
        """
    )
