"""
RiceVision AI — Phytopathology Diagnostic Bench & Benchmark Observatory
Bilingual (EN / TH) editorial scientific interface for comparing EfficientNetV2-S (Grad-CAM++)
and ViT-B/16 (14×14 Patch Attention Saliency) on 3,798 audited rice leaf specimens.
"""
import base64
import csv
import gc
import importlib
import inspect
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

# Ensure keras-hub >= 0.33.0 is installed (--no-deps bypasses tensorflow-text wheel missing on Linux cp314)
import keras_hub  # noqa: E402
if tuple(int(x) for x in keras_hub.__version__.split(".")[:2] if x.isdigit()) < (0, 30):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-q", "keras-hub==0.33.0"],
        check=False,
    )
    importlib.reload(keras_hub)

# Compatibility patch for ViTBackbone.from_config across keras-hub versions (patch_size list vs int)
try:
    from keras_hub.src.models.vit.vit_backbone import ViTBackbone
    _orig_vit_from_config = ViTBackbone.from_config

    @classmethod  # type: ignore[misc]
    def _compat_vit_from_config(cls, config):
        try:
            return _orig_vit_from_config(config)
        except TypeError:
            cfg = dict(config)
            if isinstance(cfg.get("patch_size"), (list, tuple)):
                cfg["patch_size"] = cfg["patch_size"][0]
            sig = inspect.signature(cls.__init__)
            allowed = set(sig.parameters.keys())
            if "kwargs" not in allowed:
                cfg = {k: v for k, v in cfg.items() if k in allowed}
            return cls(**cfg)

    ViTBackbone.from_config = _compat_vit_from_config
except Exception:
    pass

# ─── Page Config & Constants ──────────────────────────────────────────────────

st.set_page_config(
    page_title="RiceVision AI — Phytopathology Benchmark & XAI Bench",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

EFFNET_CHECKPOINT = MODELS_DIR / "efficientnetv2s.keras"
VIT_CHECKPOINT = MODELS_DIR / "vit_base.keras"
DEFAULT_VIT_MODEL_URL = "https://huggingface.co/ratchanonswu/ViT-B_16/resolve/main/vit_base.keras"

CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight",
]

CLASS_TH = {
    "Bacterial Leaf Blight": "โรคขอบใบแห้ง (Bacterial Leaf Blight)",
    "Brown Spot": "โรคใบจุดสีน้ำตาล (Brown Spot)",
    "Healthy Rice Leaf": "ใบข้าวปกติ (Healthy Rice Leaf)",
    "Leaf Blast": "โรคไหม้ (Leaf Blast)",
    "Leaf Scald": "โรคใบวงสีน้ำตาล (Leaf Scald)",
    "Sheath Blight": "โรคกาบใบแห้ง (Sheath Blight)",
}

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
        "pathogen": "Xanthomonas oryzae pv. oryzae",
        "en": {
            "desc": "Vascular bacterial infection prevalent in irrigated and rainfed lowland rice canopies.",
            "symptoms": "Water-soaked yellowish stripes along leaf margins progressing inward into gray-white necrotic blades with wavy edges.",
            "management": "Deploy resistant cultivars, maintain balanced nitrogen split applications, avoid field flooding, and apply copper bactericides.",
        },
        "th": {
            "desc": "โรคติดเชื้อแบคทีเรียในท่อลำเลียงน้ำของใบข้าว พบมากในนาชลประทานและนาน้ำฝนที่มีความชื้นสูง",
            "symptoms": "เกิดแผลฉ่ำน้ำสีเหลืองตามขอบใบ ลุกลามเข้าสู่กลางใบจนเปลี่ยนเป็นสีเทาขาวและขอบแผลมีลักษณะเป็นคลื่นหยัก",
            "management": "ใช้พันธุ์ข้าวต้านทาน หลีกเลี่ยงการใส่ปุ๋ยไนโตรเจนมากเกินไป ระบายน้ำในแปลงไม่ให้ท่วมขัง และใช้สารคอปเปอร์ตามคำแนะนำ",
        },
    },
    "Brown Spot": {
        "pathogen": "Cochliobolus miyabeanus (Bipolaris oryzae)",
        "en": {
            "desc": "Foliar and seed-borne fungal pathogen associated with potassium- and silicon-deficient soils.",
            "symptoms": "Oval dark-brown necrotic lesions surrounded by chlorotic yellow halos; severe infection causes unfilled panicles.",
            "management": "Restore soil potassium and silicon fertility, apply hot-water seed treatment (53–54°C, 10 min), and spray propiconazole.",
        },
        "th": {
            "desc": "โรคเชื้อราทางใบและเมล็ดพันธุ์ มักเกิดรุนแรงในดินที่ขาดธาตุโพแทสเซียมและซิลิคอน",
            "symptoms": "เกิดจุดแผลกลมรีสีน้ำตาลเข้มล้อมรอบด้วยวงสีเหลือง (Yellow Halo) หากรุนแรงจะทำให้เมล็ดลีบและน้ำหนักลดลง",
            "management": "ปรับปรุงดินโดยเพิ่มโพแทสเซียมและซิลิคอน แช่เมล็ดพันธุ์ในน้ำอุ่น 53–54°C นาน 10 นาที หรือพ่นสารโพรพิโคนาโซล",
        },
    },
    "Healthy Rice Leaf": {
        "pathogen": "Uninfected Oryza sativa specimen",
        "en": {
            "desc": "Clean foliar tissue exhibiting uniform chlorophyll density and intact epidermal margins.",
            "symptoms": "Unbroken leaf blades free of necrotic spotting, water-soaking, chlorotic striping, or fungal sporulation.",
            "management": "Maintain recommended hill spacing, intermittent irrigation, and routine canopy scouting to avoid unnecessary fungicide sprays.",
        },
        "th": {
            "desc": "ใบข้าวสมบูรณ์แข็งแรง มีสีเขียวสม่ำเสมอทั่วทั้งใบและไม่พบร่องรอยการเข้าทำลายของเชื้อสาเหตุโรค",
            "symptoms": "แผ่นใบเรียบสมบูรณ์ ไม่มีจุดแผลเน่าแห้ง แผลฉ่ำน้ำ หรือแถบสีเหลืองผิดปกติ",
            "management": "รักษาระยะปลูกให้เหมาะสม จัดการน้ำแบบเปียกสลับแห้ง และสำรวจแปลงสม่ำเสมอเพื่อลดการใช้สารเคมีที่ไม่จำเป็น",
        },
    },
    "Leaf Blast": {
        "pathogen": "Magnaporthe oryzae",
        "en": {
            "desc": "High-consequence airborne fungal disease capable of rapid canopy destruction under high humidity.",
            "symptoms": "Spindle- or diamond-shaped lesions with pale gray sporulating centers and dark reddish-brown necrotic borders.",
            "management": "Plant blast-resistant lines, avoid excessive nitrogen top-dressing, maintain flood depth, and apply tricyclazole or azoxystrobin.",
        },
        "th": {
            "desc": "โรคเชื้อราทางอากาศที่รุนแรงที่สุดชนิดหนึ่งในนาข้าว สามารถแพร่กระจายทำลายใบและคอรวงได้อย่างรวดเร็วเมื่ออากาศชื้น",
            "symptoms": "แผลรูปตาหรือรูปกระสวย (Spindle-shaped) ตรงกลางแผลสีเทาขาวและมีขอบสีน้ำตาลแดงเข้ม",
            "management": "ปลูกพันธุ์ต้านทาน งดใส่ปุ๋ยไนโตรเจนอัตราสูงในช่วงอากาศชื้น รักษาระดับน้ำในแปลง และพ่นสารไตรไซคลาโซลหรืออะซอกซีสโตรบิน",
        },
    },
    "Leaf Scald": {
        "pathogen": "Microdochium oryzae (Gerlachia oryzae)",
        "en": {
            "desc": "Fungal blight affecting mature leaves under dense canopy closure and heavy wet-season rainfall.",
            "symptoms": "Zonate oblong lesions initiating at leaf tips or margins with alternating light-tan and dark-brown bands.",
            "management": "Treat seeds with carbendazim, incorporate crop residue after harvest, avoid high seeding density, and apply benomyl.",
        },
        "th": {
            "desc": "โรคเชื้อราที่มักเกิดกับใบข้าวระยะแตกกอถึงออกรวง โดยเฉพาะในแปลงที่ปลูกหนาแน่นและมีฝนตกชุก",
            "symptoms": "แผลเริ่มต้นที่ปลายใบหรือขอบใบ ลักษณะเป็นวงซ้อนสลับสีน้ำตาลอ่อนและน้ำตาลเข้มคล้ายรอยน้ำร้อนลวก",
            "management": "คลุกเมล็ดพันธุ์ด้วยคาร์เบนดาซิม ไถกลบตอซังหลังเก็บเกี่ยว ลดความหนาแน่นของต้นข้าว และพ่นสารป้องกันกำจัดเชื้อราเมื่อจำเป็น",
        },
    },
    "Sheath Blight": {
        "pathogen": "Rhizoctonia solani AG-1 IA",
        "en": {
            "desc": "Soil-borne sclerotial fungus initiating near the waterline and climbing the upper foliar canopy.",
            "symptoms": "Irregular greenish-gray water-soaked lesions coalescing into characteristic snake-skin banding with brown margins.",
            "management": "Optimize plant spacing, drain paddies mid-season to reduce canopy humidity, and apply validamycin at early booting.",
        },
        "th": {
            "desc": "โรคเชื้อราในดินที่เริ่มเข้าทำลายบริเวณกาบใบใกล้ผิวน้ำ ก่อนลุกลามขึ้นสู่ใบยอดในทรงพุ่มที่อับชื้น",
            "symptoms": "แผลฉ่ำน้ำสีเขียวปนเทารูปวงรีไม่สม่ำเสมอ ขยายตัวรวมกันเป็นลวดลายคล้ายหนังงูและมีขอบแผลสีน้ำตาล",
            "management": "จัดระยะปลูกให้โปร่ง ระบายน้ำออกจากแปลงเป็นระยะเพื่อลดความชื้นในทรงพุ่ม และพ่นสารวาลิดามัยซินหรืออะซอกซีสโตรบิน",
        },
    },
}

LEAF_MARK_SVG = """
<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="28" height="28" rx="7" fill="#143323"/>
  <path d="M7.5 20.5C7.5 13.5 12.5 7.5 20.5 7.5C20.5 15.5 14.5 20.5 7.5 20.5Z" stroke="#A7F3D0" stroke-width="1.75" stroke-linejoin="round"/>
  <path d="M7.5 20.5L16.5 11.5" stroke="#A7F3D0" stroke-width="1.5" stroke-linecap="round"/>
</svg>
"""

# ─── Custom Editorial Scientific Styling ──────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Thai:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Source+Sans+3:wght@400;500;600;700&display=swap');

    :root {
        --bg-paper: #F7F6F2;
        --bg-surface: #FFFFFF;
        --bg-subtle: #EFECE6;
        --ink-primary: #14261C;
        --ink-secondary: #3D5245;
        --ink-muted: #5C6E63;
        --rule-hairline: #DDD9D0;
        --eff-green: #1B5E3A;
        --eff-surface: #F1F7F3;
        --eff-border: #B8D8C5;
        --vit-violet: #5B2E91;
        --vit-surface: #F6F2FA;
        --vit-border: #D4C2EC;
    }

    ::selection {
        background: #1B5E3A;
        color: #FFFFFF;
    }

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-paper);
    }
    ::-webkit-scrollbar-thumb {
        background: #C5C0B5;
        border-radius: 4px;
    }

    .stApp {
        background-color: var(--bg-paper);
        color: var(--ink-primary);
        font-family: 'Source Sans 3', 'IBM Plex Sans Thai', -apple-system, BlinkMacSystemFont, sans-serif;
        font-variant-numeric: tabular-nums;
    }

    .block-container {
        padding-top: 3.75rem !important;
        padding-bottom: 3.5rem;
        max-width: 1180px;
    }

    header[data-testid="stHeader"] {
        background: rgba(247, 246, 242, 0.88);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid var(--rule-hairline);
    }
    #MainMenu, footer, [data-testid="stHeaderActionElements"] {
        display: none !important;
        visibility: hidden !important;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-surface) !important;
        border-right: 1px solid var(--rule-hairline) !important;
    }

    h1, h2, h3, .rv-serif {
        font-family: 'Newsreader', 'IBM Plex Sans Thai', Georgia, serif !important;
        letter-spacing: -0.02em;
        color: var(--ink-primary) !important;
        font-weight: 500 !important;
    }

    .rv-mono {
        font-family: 'IBM Plex Mono', monospace;
        font-variant-numeric: tabular-nums;
    }

    .rv-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .rv-brand-title {
        font-family: 'Newsreader', 'IBM Plex Sans Thai', Georgia, serif;
        font-size: 1.55rem;
        line-height: 1.05;
        color: var(--ink-primary);
        letter-spacing: -0.02em;
    }
    .rv-brand-sub {
        font-size: 0.76rem;
        color: var(--ink-secondary);
    }

    div[data-testid="stRadio"] > div[role="radiogroup"] {
        background: var(--bg-surface);
        padding: 0.3rem 0.45rem;
        border-radius: 10px;
        border: 1px solid var(--rule-hairline);
        gap: 0.35rem;
    }

    .rv-lead-box {
        background: var(--bg-surface);
        border: 1px solid var(--rule-hairline);
        border-radius: 14px;
        padding: 2rem 2.25rem;
        margin-bottom: 1.25rem;
    }
    .rv-lead-title {
        font-family: 'Newsreader', 'IBM Plex Sans Thai', Georgia, serif;
        font-size: 2.35rem;
        line-height: 1.15;
        letter-spacing: -0.02em;
        color: var(--ink-primary);
        margin: 0 0 0.75rem 0;
        max-width: 32ch;
    }
    .rv-lead-copy {
        font-size: 0.96rem;
        line-height: 1.68;
        color: var(--ink-secondary);
        max-width: 72ch;
        margin: 0;
    }

    .rv-ledger {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        background: var(--bg-surface);
        border: 1px solid var(--rule-hairline);
        border-radius: 14px;
        overflow: hidden;
        margin-bottom: 1.5rem;
    }
    .rv-ledger-cell {
        padding: 1.15rem 1.35rem;
        border-right: 1px solid var(--rule-hairline);
    }
    .rv-ledger-cell:last-child {
        border-right: none;
    }
    .rv-ledger-val {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--ink-primary);
        line-height: 1.15;
        font-variant-numeric: tabular-nums;
    }
    .rv-ledger-val-green { color: var(--eff-green); }
    .rv-ledger-val-violet { color: var(--vit-violet); }
    .rv-ledger-name {
        font-size: 0.84rem;
        font-weight: 600;
        color: var(--ink-primary);
        margin-top: 0.3rem;
    }
    .rv-ledger-note {
        font-size: 0.76rem;
        color: var(--ink-muted);
        margin-top: 0.1rem;
    }

    .rv-dossier {
        background: var(--bg-surface);
        border: 1px solid var(--rule-hairline);
        border-radius: 14px;
        padding: 1.5rem 1.65rem;
        height: 100%;
    }
    .rv-dossier-eff {
        background: var(--eff-surface);
        border-color: var(--eff-border);
    }
    .rv-dossier-vit {
        background: var(--vit-surface);
        border-color: var(--vit-border);
    }
    .rv-dossier h3 {
        font-size: 1.5rem !important;
        margin: 0 0 0.4rem 0 !important;
    }
    .rv-dossier p {
        font-size: 0.9rem;
        line-height: 1.6;
        color: var(--ink-secondary);
        max-width: 65ch;
        margin: 0 0 1rem 0;
    }
    .rv-spec-row {
        display: flex;
        justify-content: space-between;
        padding: 0.45rem 0;
        border-top: 1px solid rgba(20, 38, 28, 0.09);
        font-size: 0.84rem;
    }
    .rv-spec-key { color: var(--ink-secondary); font-weight: 500; }
    .rv-spec-val {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--ink-primary);
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }

    .rv-banner-consensus {
        background: var(--eff-surface);
        border: 1px solid var(--eff-border);
        color: var(--ink-primary);
        padding: 0.85rem 1.15rem;
        border-radius: 12px;
        font-size: 0.88rem;
        margin: 0.75rem 0 1rem 0;
    }
    .rv-banner-divergence {
        background: #FDF6EC;
        border: 1px solid #E9D2B0;
        color: #4A3212;
        padding: 0.85rem 1.15rem;
        border-radius: 12px;
        font-size: 0.88rem;
        margin: 0.75rem 0 1rem 0;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-surface);
        border: 1px solid var(--rule-hairline) !important;
        border-radius: 14px !important;
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
    if not ckpt.exists() and model_key == "vit_base":
        vit_url = os.environ.get("VIT_MODEL_URL")
        if not vit_url:
            try:
                vit_url = st.secrets.get("VIT_MODEL_URL", None)
            except Exception:
                vit_url = None
        vit_url = (vit_url or DEFAULT_VIT_MODEL_URL).replace("/blob/", "/resolve/")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(vit_url, str(ckpt))
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {ckpt.name}. "
            "Set VIT_MODEL_URL in Streamlit Secrets or select EfficientNetV2-S Only."
        )
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
        xai_method = "Attention Saliency (14×14)"

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


def style_plotly(fig: go.Figure, height: int = 240) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3, IBM Plex Sans Thai, sans-serif", color="#14261C", size=12),
        margin=dict(l=6, r=14, t=24, b=10),
    )
    fig.update_xaxes(gridcolor="#E6E4DD", zerolinecolor="#E6E4DD")
    fig.update_yaxes(gridcolor="#E6E4DD", zerolinecolor="#E6E4DD")
    return fig


# ─── Left Sidebar Navigation, Language Toggle & Runtime Control ───────────────

with st.sidebar:
    st.markdown(
        f'<div class="rv-brand" style="padding:0.25rem 0 0.9rem 0;border-bottom:1px solid #DDD9D0;margin-bottom:0.85rem;">'
        f'{LEAF_MARK_SVG}'
        f'<div><div class="rv-brand-title">RiceVision AI</div>'
        f'<div class="rv-brand-sub">EfficientNetV2-S &amp; ViT-B/16</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    lang_choice = st.radio(
        "Language / ภาษา",
        ["TH — ภาษาไทย", "EN — English"],
        index=0,
        horizontal=True,
    )
    is_th = lang_choice.startswith("TH")

    nav_keys = ["overview", "bench", "benchmark", "atlas", "methodology"]
    nav_labels = {
        "overview": "ภาพรวมโครงการ (Overview)" if is_th else "Overview",
        "bench": "วิเคราะห์และทำนายโรค (Diagnostic Bench)" if is_th else "Diagnostic Bench",
        "benchmark": "เปรียบเทียบโมเดล (Benchmark)" if is_th else "Benchmark",
        "atlas": "ฐานข้อมูลโรคใบข้าว 6 ชนิด (Pathology Atlas)" if is_th else "Pathology Atlas",
        "methodology": "กระบวนการวิจัย (Methodology)" if is_th else "Methodology",
    }
    page_key = st.radio(
        "Navigation",
        nav_keys,
        format_func=lambda k: nav_labels[k],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border:none;border-top:1px solid #DDD9D0;margin:1.1rem 0;"/>', unsafe_allow_html=True)
    st.markdown("**สถานะโมเดล (Runtime Checkpoints)**" if is_th else "**Runtime Checkpoints**")
    st.caption(f"EfficientNetV2-S (83.0 MB): {'พร้อมใช้งาน' if (EFFNET_CHECKPOINT.exists() and is_th) else ('Ready' if EFFNET_CHECKPOINT.exists() else 'Missing')}")
    st.caption(f"ViT-B/16 (982.5 MB): {'พร้อมใช้งาน' if (VIT_CHECKPOINT.exists() and is_th) else ('Ready' if VIT_CHECKPOINT.exists() else 'HF Hub Auto-Load')}")
    if st.button("ล้างแคชโมเดลใน RAM" if is_th else "Release Cached Models from RAM", use_container_width=True):
        load_model_cached.clear()
        tf.keras.backend.clear_session()
        gc.collect()
        st.success("ล้างแคชโมเดลออกจากหน่วยความจำแล้ว" if is_th else "Cleared model weights from memory.")


DISCLAIMER_TEXT = (
    "ผลการทำนาย ภาพ Grad-CAM++ และ ViT 14×14 Attention Saliency เป็นผลลัพธ์จากการวิจัย Deep Learning "
    "เพื่อช่วยสนับสนุนการตัดสินใจเบื้องต้น ควรตรวจสอบร่วมกับผู้เชี่ยวชาญโรคพืชก่อนนำไปใช้จริงในแปลงนา"
    if is_th else
    "Predictions, Grad-CAM++, and ViT 14×14 Attention Saliency maps are research outputs intended for "
    "agronomic decision-support. Confirm field diagnoses with certified plant pathologists before treatment."
)


# ─── Pages ────────────────────────────────────────────────────────────────────

if page_key == "overview":
    if is_th:
        st.markdown(
            """
            <div class="rv-lead-box">
                <h1 class="rv-lead-title">การจำแนกโรคใบข้าวด้วยโครงข่าย Convolutional และ Vision Transformer</h1>
                <p class="rv-lead-copy">
                    งานวิจัยเชิงประจักษ์เปรียบเทียบโมเดลโครงข่ายประสาทเทียมขนาดเล็ก
                    (<strong>EfficientNetV2-S</strong>) กับโมเดล <strong>Vision Transformer (ViT-B/16)</strong>
                    ที่ใช้กลไก Self-Attention มองภาพรวมทั้งใบข้าว บนชุดข้อมูลภาพใบข้าว 3,798 ภาพ (6 คลาส)
                    ที่ผ่านการคัดกรองภาพซ้ำด้วย SHA-256 และ pHash แบบไร้ Data Leakage พร้อมอธิบายผลด้วย
                    Grad-CAM++ และ 14×14 Patch Attention Saliency
                </p>
            </div>

            <div class="rv-ledger">
                <div class="rv-ledger-cell">
                    <div class="rv-ledger-val rv-ledger-val-violet">97.72%</div>
                    <div class="rv-ledger-name">ความแม่นยำของ ViT-B/16</div>
                    <div class="rv-ledger-note">สูงกว่า CNN +2.46% (ชุดทดสอบ 570 ภาพ)</div>
                </div>
                <div class="rv-ledger-cell">
                    <div class="rv-ledger-val rv-ledger-val-green">95.26%</div>
                    <div class="rv-ledger-name">ความแม่นยำ EfficientNetV2-S</div>
                    <div class="rv-ledger-note">ขนาดเพียง 83.04 MB (เบากว่า 11.8 เท่า)</div>
                </div>
                <div class="rv-ledger-cell">
                    <div class="rv-ledger-val">130.8 FPS</div>
                    <div class="rv-ledger-name">ความเร็วประมวลผล ViT (GPU)</div>
                    <div class="rv-ledger-note">7.64 ms/ภาพ เทียบกับ 14.88 ms ของ CNN</div>
                </div>
                <div class="rv-ledger-cell">
                    <div class="rv-ledger-val">3,798</div>
                    <div class="rv-ledger-name">ภาพใบข้าวหลังคัดกรองซ้ำ</div>
                    <div class="rv-ledger-note">ตัดซ้ำ SHA-256 31 ภาพ • จัดกลุ่ม pHash 67 กลุ่ม</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="rv-lead-box">
                <h1 class="rv-lead-title">Foliar Rice Disease Classification Across Convolutional and Self-Attention Architectures</h1>
                <p class="rv-lead-copy">
                    A controlled empirical study evaluating a lightweight mobile convolutional network
                    (<strong>EfficientNetV2-S</strong>) against a global self-attention Vision Transformer
                    (<strong>ViT-B/16</strong>) on 3,798 cryptographic and perceptual hash–deduplicated
                    rice leaf specimens across six disease classes, verified with pre-softmax Grad-CAM++
                    and 14×14 patch input-gradient saliency.
                </p>
            </div>

            <div class="rv-ledger">
                <div class="rv-ledger-cell">
                    <div class="rv-ledger-val rv-ledger-val-violet">97.72%</div>
                    <div class="rv-ledger-name">ViT-B/16 Test Accuracy</div>
                    <div class="rv-ledger-note">+2.46% over CNN baseline (570 test leaves)</div>
                </div>
                <div class="rv-ledger-cell">
                    <div class="rv-ledger-val rv-ledger-val-green">95.26%</div>
                    <div class="rv-ledger-name">EfficientNetV2-S Accuracy</div>
                    <div class="rv-ledger-note">83.04 MB checkpoint (11.8× lighter)</div>
                </div>
                <div class="rv-ledger-cell">
                    <div class="rv-ledger-val">130.8 FPS</div>
                    <div class="rv-ledger-name">ViT-B/16 GPU Throughput</div>
                    <div class="rv-ledger-note">7.64 ms/image vs. 14.88 ms on CNN</div>
                </div>
                <div class="rv-ledger-cell">
                    <div class="rv-ledger-val">3,798</div>
                    <div class="rv-ledger-name">Zero-Leakage Specimens</div>
                    <div class="rv-ledger-note">31 SHA-256 duplicates removed • 67 pHash groups</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"""
            <div class="rv-dossier rv-dossier-eff">
                <h3>EfficientNetV2-S</h3>
                <p>
                    {"โครงข่าย CNN ยุคใหม่ที่ใช้ Fused-MBConv ออกแบบมาให้มีขนาดไฟล์กะทัดรัด ประหยัด RAM และจับขอบเขตรอยโรคเฉพาะจุดได้อย่างแม่นยำ"
                     if is_th else
                     "Fused-MBConv convolutional architecture optimized for edge deployment and low memory overhead, isolating sharp local necrotic lesion boundaries."}
                </p>
                <div class="rv-spec-row"><span class="rv-spec-key">{"ความแม่นยำ / Macro F1" if is_th else "Test Accuracy / Macro F1"}</span><span class="rv-spec-val">95.26% / 95.21%</span></div>
                <div class="rv-spec-row"><span class="rv-spec-key">{"Precision / Recall เฉลี่ย" if is_th else "Macro Precision / Recall"}</span><span class="rv-spec-val">95.23% / 95.22%</span></div>
                <div class="rv-spec-row"><span class="rv-spec-key">ROC-AUC (One-vs-Rest)</span><span class="rv-spec-val">0.9961</span></div>
                <div class="rv-spec-row"><span class="rv-spec-key">{"พารามิเตอร์ & ขนาดไฟล์" if is_th else "Parameters & Size"}</span><span class="rv-spec-val">20,660,838 • 83.04 MB</span></div>
                <div class="rv-spec-row"><span class="rv-spec-key">{"เทคนิคอธิบายผล (XAI)" if is_th else "Visual Attribution"}</span><span class="rv-spec-val">Pre-softmax Grad-CAM++</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"""
            <div class="rv-dossier rv-dossier-vit">
                <h3>ViT-B/16 (Vision Transformer)</h3>
                <p>
                    {"โครงข่าย Transformer 12 ชั้นที่ใช้ Multi-Head Self-Attention แบ่งภาพเป็น 196 Patches (16×16) เพื่อเชื่อมโยงบริบททั่วทั้งใบข้าวและลดปัญหาการจำฉากหลัง"
                     if is_th else
                     "12-layer Multi-Head Self-Attention encoder operating on 196 non-overlapping 16×16 patches, capturing full-blade context to suppress background weed shortcut bias."}
                </p>
                <div class="rv-spec-row"><span class="rv-spec-key">{"ความแม่นยำ / Macro F1" if is_th else "Test Accuracy / Macro F1"}</span><span class="rv-spec-val">97.72% / 97.71%</span></div>
                <div class="rv-spec-row"><span class="rv-spec-key">{"Precision / Recall เฉลี่ย" if is_th else "Macro Precision / Recall"}</span><span class="rv-spec-val">97.80% / 97.70%</span></div>
                <div class="rv-spec-row"><span class="rv-spec-key">ROC-AUC (One-vs-Rest)</span><span class="rv-spec-val">0.9974</span></div>
                <div class="rv-spec-row"><span class="rv-spec-key">{"พารามิเตอร์ & ขนาดไฟล์" if is_th else "Parameters & Size"}</span><span class="rv-spec-val">85,803,270 • 982.52 MB</span></div>
                <div class="rv-spec-row"><span class="rv-spec-key">{"เทคนิคอธิบายผล (XAI)" if is_th else "Visual Attribution"}</span><span class="rv-spec-val">14×14 Patch Attention Saliency</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


elif page_key == "bench":
    st.markdown("## ระบบวิเคราะห์และทำนายโรคใบข้าวพร้อม Explainable AI" if is_th else "## Foliar Diagnostic Bench & Explainable AI")
    st.caption(
        "อัปโหลดภาพใบข้าวเพื่อทำนายผลสด หรือเลือกดูตัวอย่างจากชุดข้อมูลทดสอบทั้ง 6 คลาสพร้อมเปรียบเทียบ Grad-CAM++ และ 14×14 Attention Saliency"
        if is_th else
        "Evaluate custom field specimens or inspect pre-computed test leaves with side-by-side Grad-CAM++ and 14×14 Attention Saliency."
    )

    tab_live, tab_sample = st.tabs(
        ["อัปโหลดภาพทำนายสด (Live Inference)", "แกลเลอรีภาพตัวอย่าง 6 คลาส (Test Gallery)"]
        if is_th else
        ["Upload Specimen (Live Inference)", "Audited 6-Class Test Gallery"]
    )

    with tab_live:
        with st.container(border=True):
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2.1, 1.4, 1.5])
            with col_ctrl1:
                uploaded = st.file_uploader(
                    "เลือกไฟล์ภาพใบข้าว (.jpg, .png, .webp)" if is_th else "Select Rice Leaf Specimen (.jpg, .png, .webp)",
                    type=["jpg", "jpeg", "png", "webp"],
                )
            with col_ctrl2:
                mode_opts = (
                    ["รันเปรียบเทียบทั้ง 2 โมเดล", "EfficientNetV2-S เท่านั้น (ประหยัด RAM)", "ViT-B/16 เท่านั้น"]
                    if is_th else
                    ["Both Architectures (Side-by-Side)", "EfficientNetV2-S Only (Low RAM)", "ViT-B/16 Only"]
                )
                model_choice = st.selectbox("โหมดการรันโมเดล" if is_th else "Architecture Execution Mode", mode_opts)
            with col_ctrl3:
                alpha = st.slider("ความเข้ม Heatmap (α)" if is_th else "Attribution Overlay Opacity (α)", 0.15, 0.85, 0.45, 0.05)
                view_mode = st.radio(
                    "รูปแบบการแสดงภาพ" if is_th else "Layer View",
                    ["Overlay", "Raw Heatmap", "Original"],
                    horizontal=True,
                )

        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            if st.button("เริ่มวิเคราะห์โรคใบข้าว" if is_th else "Execute Foliar Diagnosis", type="primary", use_container_width=True):
                idx_mode = mode_opts.index(model_choice)
                targets = ["efficientnetv2s", "vit_base"] if idx_mode == 0 else (["efficientnetv2s"] if idx_mode == 1 else ["vit_base"])
                results = {}
                for mk in targets:
                    label = "EfficientNetV2-S" if mk == "efficientnetv2s" else "ViT-B/16"
                    with st.spinner(f"กำลังประมวลผลด้วย {label}..." if is_th else f"Running {label} inference and attribution..."):
                        try:
                            results[mk] = run_inference(pil_img, mk, alpha)
                        except FileNotFoundError as exc:
                            st.warning(f"{label}: {exc}")
                        except Exception as exc:
                            st.error(f"{label} error: {exc}")

                if len(results) == 2:
                    e_cls = results["efficientnetv2s"]["predicted_class"]
                    v_cls = results["vit_base"]["predicted_class"]
                    if e_cls == v_cls:
                        st.markdown(
                            f'<div class="rv-banner-consensus"><strong>{"ผลลัพธ์ตรงกัน (Consensus):" if is_th else "Architectural Consensus:"}</strong> '
                            f'{"ทั้งสองโมเดลทำนายตรงกันว่าเป็น" if is_th else "Both EfficientNetV2-S and ViT-B/16 classify this specimen as"} '
                            f'<strong>{CLASS_TH[v_cls] if is_th else v_cls}</strong></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="rv-banner-divergence"><strong>{"ผลลัพธ์ต่างกัน (Divergence):" if is_th else "Architectural Divergence:"}</strong> '
                            f'EfficientNetV2-S → <strong>{CLASS_TH[e_cls] if is_th else e_cls}</strong> | '
                            f'ViT-B/16 → <strong>{CLASS_TH[v_cls] if is_th else v_cls}</strong></div>',
                            unsafe_allow_html=True,
                        )

                if results:
                    cols = st.columns(len(results))
                    for col, (mk, res) in zip(cols, results.items()):
                        title = "EfficientNetV2-S — Grad-CAM++" if mk == "efficientnetv2s" else "ViT-B/16 — Attention Saliency"
                        bar_color = "#1B5E3A" if mk == "efficientnetv2s" else "#5B2E91"
                        with col:
                            with st.container(border=True):
                                st.subheader(title)
                                img_to_show = (
                                    res["overlay_img"] if view_mode == "Overlay"
                                    else (res["raw_hm_img"] if view_mode == "Raw Heatmap" else res["orig_img"])
                                )
                                st.image(img_to_show, use_container_width=True, caption=f"{res['xai_method']} • {res['inference_ms']} ms")
                                disp_cls = CLASS_TH[res["predicted_class"]] if is_th else res["predicted_class"]
                                st.metric("ผลการวินิจฉัย" if is_th else "Predicted Diagnosis", disp_cls, f"{res['confidence']*100:.2f}% confidence")
                                score_df = pd.DataFrame({
                                    "Class": list(res["scores"].keys()),
                                    "Probability (%)": [v * 100 for v in res["scores"].values()],
                                }).sort_values("Probability (%)", ascending=True)
                                fig = px.bar(
                                    score_df, x="Probability (%)", y="Class", orientation="h",
                                    color_discrete_sequence=[bar_color],
                                )
                                fig.update_layout(xaxis_range=[0, 100])
                                st.plotly_chart(style_plotly(fig, 225), use_container_width=True)
                    st.caption(DISCLAIMER_TEXT)

    with tab_sample:
        samples = load_json(str(OUTPUTS_DIR / "sample_results.json"))
        class_imgs = load_json(str(OUTPUTS_DIR / "class_images.json"))
        if not samples:
            st.warning("Pre-computed gallery file `outputs/sample_results.json` not found.")
        else:
            scol1, scol2 = st.columns([2.2, 1])
            with scol1:
                chosen = st.selectbox(
                    "เลือกตัวอย่างภาพใบข้าวจากชุดทดสอบ" if is_th else "Select Audited Reference Specimen",
                    list(samples.keys()),
                    format_func=lambda k: CLASS_TH.get(k, k) if is_th else k,
                )
            with scol2:
                sample_hm_view = st.radio(
                    "เลเยอร์ Heatmap" if is_th else "Attribution Layer",
                    ["Overlay", "Raw Heatmap"],
                    horizontal=True,
                )

            item = samples[chosen]
            eff = item["efficientnetv2s"]
            vit = item["vit_base"]
            hm_key = "gradcam_overlay" if sample_hm_view == "Overlay" else "gradcam_raw"

            if eff["predicted_class"] == vit["predicted_class"]:
                st.markdown(
                    f'<div class="rv-banner-consensus"><strong>{"ผลลัพธ์ตรงกัน (Consensus):" if is_th else "Architectural Consensus:"}</strong> '
                    f'{"ทั้งสองโมเดลทายว่าเป็น" if is_th else "Both models predict"} <strong>{vit["predicted_class"]}</strong> '
                    f'({"คลาสจริง" if is_th else "Ground Truth"}: <strong>{chosen}</strong>)</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="rv-banner-divergence"><strong>{"ผลลัพธ์ต่างกัน (Divergence):" if is_th else "Architectural Divergence:"}</strong> '
                    f'EfficientNetV2-S → <strong>{eff["predicted_class"]}</strong> | ViT-B/16 → <strong>{vit["predicted_class"]}</strong> '
                    f'({"คลาสจริง" if is_th else "Ground Truth"}: <strong>{chosen}</strong>)</div>',
                    unsafe_allow_html=True,
                )

            c1, c2, c3 = st.columns(3)
            with c1:
                with st.container(border=True):
                    st.subheader("ภาพใบข้าวต้นฉบับ" if is_th else "Original Specimen")
                    orig_b64 = item.get("original_image") or class_imgs.get(chosen)
                    if orig_b64:
                        st.image(b64_to_pil(orig_b64), caption=f"Ground Truth: {chosen}", use_container_width=True)
                    else:
                        st.info("Reference image unavailable.")

            for col, (res, title, xai_label, bar_color) in zip([c2, c3], [
                (eff, "EfficientNetV2-S", "Grad-CAM++", "#1B5E3A"),
                (vit, "ViT-B/16 (Transformer)", "Attention Saliency (14×14)", "#5B2E91"),
            ]):
                with col:
                    with st.container(border=True):
                        st.subheader(title)
                        st.image(
                            b64_to_pil(res[hm_key]),
                            caption=f"{xai_label} • {res['inference_ms']} ms",
                            use_container_width=True,
                        )
                        st.metric("ผลการทำนาย" if is_th else "Predicted Diagnosis", res["predicted_class"], f"{res['confidence']*100:.2f}% confidence")
                        score_df = pd.DataFrame({
                            "Class": list(res["scores"].keys()),
                            "Probability (%)": [v * 100 for v in res["scores"].values()],
                        }).sort_values("Probability (%)", ascending=True)
                        fig = px.bar(
                            score_df, x="Probability (%)", y="Class", orientation="h",
                            color_discrete_sequence=[bar_color],
                        )
                        fig.update_layout(xaxis_range=[0, 100])
                        st.plotly_chart(style_plotly(fig, 215), use_container_width=True)


elif page_key == "benchmark":
    st.markdown("## ผลการเปรียบเทียบประสิทธิภาพโมเดล (Benchmark)" if is_th else "## Head-to-Head Empirical Benchmark")
    st.caption(
        "ประเมินผลบนชุดข้อมูลทดสอบ 570 ภาพที่ไม่เคยผ่านการเทรนมาก่อน ภายใต้การคัดกรองภาพซ้ำด้วย SHA-256 และ pHash"
        if is_th else
        "Evaluated on 570 held-out test specimens under identical SHA-256 + pHash zero-leakage protocols."
    )

    eff_m = load_json(str(OUTPUTS_DIR / "efficientnetv2s" / "metrics.json"))
    vit_m = load_json(str(OUTPUTS_DIR / "vit_base" / "metrics.json"))
    eff_e = load_json(str(OUTPUTS_DIR / "efficientnetv2s" / "efficiency.json"))
    vit_e = load_json(str(OUTPUTS_DIR / "vit_base" / "efficiency.json"))

    tab1, tab2, tab3 = st.tabs(
        ["ตารางเปรียบเทียบภาพรวม", "คะแนน F1 รายคลาส (6 โรค)", "ตาราง Confusion Matrix"]
        if is_th else
        ["Aggregate & Efficiency Ledger", "Per-Class F1 Breakdown", "Confusion Matrices"]
    )
    with tab1:
        left_col, right_col = st.columns([1.25, 1])
        with left_col:
            with st.container(border=True):
                st.subheader("ตารางตัวชี้วัดความแม่นยำและทรัพยากร" if is_th else "Classification & Computational Ledger")
                comp_df = pd.DataFrame([
                    {"Metric": "Test Accuracy", "EfficientNetV2-S": f"{eff_m.get('accuracy', 0.9526)*100:.2f}%", "ViT-B/16": f"{vit_m.get('accuracy', 0.9772)*100:.2f}%", "Advantage": "ViT-B/16 (+2.46%)"},
                    {"Metric": "Macro Precision", "EfficientNetV2-S": f"{eff_m.get('precision_macro', 0.9523)*100:.2f}%", "ViT-B/16": f"{vit_m.get('precision_macro', 0.9780)*100:.2f}%", "Advantage": "ViT-B/16 (+2.57%)"},
                    {"Metric": "Macro Recall", "EfficientNetV2-S": f"{eff_m.get('recall_macro', 0.9522)*100:.2f}%", "ViT-B/16": f"{vit_m.get('recall_macro', 0.9770)*100:.2f}%", "Advantage": "ViT-B/16 (+2.48%)"},
                    {"Metric": "Macro F1-Score", "EfficientNetV2-S": f"{eff_m.get('f1_score_macro', 0.9521)*100:.2f}%", "ViT-B/16": f"{vit_m.get('f1_score_macro', 0.9771)*100:.2f}%", "Advantage": "ViT-B/16 (+2.50%)"},
                    {"Metric": "ROC-AUC (Macro OvR)", "EfficientNetV2-S": f"{eff_m.get('roc_auc_macro', 0.9961)*100:.2f}%", "ViT-B/16": f"{vit_m.get('roc_auc_macro', 0.9974)*100:.2f}%", "Advantage": "ViT-B/16 (+0.13%)"},
                    {"Metric": "Checkpoint Size", "EfficientNetV2-S": "83.04 MB", "ViT-B/16": "982.52 MB", "Advantage": "EfficientNetV2-S (11.8× smaller)"},
                    {"Metric": "Total Parameters", "EfficientNetV2-S": f"{eff_e.get('total_parameters', 20660838):,}", "ViT-B/16": f"{vit_e.get('total_parameters', 85803270):,}", "Advantage": "EfficientNetV2-S (4.15× fewer)"},
                    {"Metric": "Training Time", "EfficientNetV2-S": "9.98 min", "ViT-B/16": "4.75 min", "Advantage": "ViT-B/16 (2.1× faster)"},
                    {"Metric": "Avg GPU Latency", "EfficientNetV2-S": f"{eff_e.get('average_latency_ms', 14.88)} ms", "ViT-B/16": f"{vit_e.get('average_latency_ms', 7.64)} ms", "Advantage": "ViT-B/16 (1.95× faster)"},
                    {"Metric": "GPU Throughput", "EfficientNetV2-S": f"{eff_e.get('throughput_fps', 67.2)} fps", "ViT-B/16": f"{vit_e.get('throughput_fps', 130.8)} fps", "Advantage": "ViT-B/16 (+94.6%)"},
                ])
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

        with right_col:
            with st.container(border=True):
                st.subheader("กราฟเรดาร์เปรียบเทียบประสิทธิภาพ" if is_th else "Macro Metric Profile")
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
                fig_radar.add_trace(go.Scatterpolar(r=eff_vals + [eff_vals[0]], theta=radar_cats + [radar_cats[0]], fill="toself", name="EfficientNetV2-S", line_color="#1B5E3A"))
                fig_radar.add_trace(go.Scatterpolar(r=vit_vals + [vit_vals[0]], theta=radar_cats + [radar_cats[0]], fill="toself", name="ViT-B/16", line_color="#5B2E91"))
                fig_radar.update_layout(
                    polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[92, 100], gridcolor="#DDD9D0")),
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=370,
                    margin=dict(l=30, r=30, t=20, b=20),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        df_eff = load_classification_report(str(OUTPUTS_DIR / "efficientnetv2s" / "classification_report.csv"))
        df_vit = load_classification_report(str(OUTPUTS_DIR / "vit_base" / "classification_report.csv"))
        if not df_eff.empty and not df_vit.empty:
            merged = pd.DataFrame({
                "Class": df_eff["Class"],
                "EfficientNetV2-S F1 (%)": df_eff["F1-Score"].round(2),
                "ViT-B/16 F1 (%)": df_vit["F1-Score"].round(2),
                "Delta (ViT - CNN)": (df_vit["F1-Score"] - df_eff["F1-Score"]).round(2),
                "Test Support": df_eff["Support"],
            })
            with st.container(border=True):
                st.subheader("เปรียบเทียบ F1-Score ทั้ง 6 คลาสโรคใบข้าว" if is_th else "Per-Class F1-Score Comparison")
                fig = go.Figure(data=[
                    go.Bar(name="EfficientNetV2-S", x=merged["Class"], y=merged["EfficientNetV2-S F1 (%)"], marker_color="#1B5E3A"),
                    go.Bar(name="ViT-B/16", x=merged["Class"], y=merged["ViT-B/16 F1 (%)"], marker_color="#5B2E91"),
                ])
                fig.update_layout(barmode="group", yaxis_range=[88, 100])
                st.plotly_chart(style_plotly(fig, 360), use_container_width=True)
                st.dataframe(merged, use_container_width=True, hide_index=True)

    with tab3:
        c1, c2 = st.columns(2)
        for col, (mdir, title, scale) in zip([c1, c2], [
            ("efficientnetv2s", "EfficientNetV2-S Confusion Matrix (n = 570)", "Greens"),
            ("vit_base", "ViT-B/16 Confusion Matrix (n = 570)", "Purples"),
        ]):
            cm = load_confusion_matrix(mdir)
            if not cm.empty:
                with col:
                    with st.container(border=True):
                        st.subheader(title)
                        fig = px.imshow(cm, text_auto=True, color_continuous_scale=scale)
                        st.plotly_chart(style_plotly(fig, 410), use_container_width=True)


elif page_key == "atlas":
    st.markdown("## ฐานข้อมูลโรคใบข้าวทั้ง 6 คลาส (Foliar Pathology Atlas)" if is_th else "## Foliar Rice Pathology Atlas")
    st.caption(
        "ข้อมูลเชื้อสาเหตุ ลักษณะอาการของรอยโรคบนใบข้าว และแนวทางการจัดการดูแลแปลงนา"
        if is_th else
        "Diagnostic reference for the six target Oryza sativa foliar categories, pathogen etiology, and agronomic management."
    )
    class_imgs = load_json(str(OUTPUTS_DIR / "class_images.json"))
    cols = st.columns(2)
    for i, cls in enumerate(CLASS_NAMES):
        info = DISEASE_INFO[cls]
        lang_block = info["th"] if is_th else info["en"]
        with cols[i % 2]:
            with st.container(border=True):
                c_img, c_txt = st.columns([1, 2.1])
                with c_img:
                    if cls in class_imgs and class_imgs[cls]:
                        st.image(b64_to_pil(class_imgs[cls]), use_container_width=True)
                with c_txt:
                    st.subheader(CLASS_TH[cls] if is_th else cls)
                    st.caption(f"{'เชื้อสาเหตุ' if is_th else 'Causal Agent'}: {info['pathogen']}")
                    st.write(lang_block["desc"])
                    st.markdown(f"**{'อาการของโรค' if is_th else 'Visual Symptoms'}:** {lang_block['symptoms']}")
                    st.markdown(f"**{'การป้องกันและจัดการ' if is_th else 'Field Management'}:** {lang_block['management']}")


elif page_key == "methodology":
    st.markdown("## กระบวนการวิจัยและสถาปัตยกรรมระบบ (Methodology)" if is_th else "## Experimental Protocol & Pipeline Architecture")
    st.caption(
        "ลำดับการดำเนินงานตั้งแต่การคัดกรองภาพซ้ำ การปรับจูนโมเดล 2 ระยะ โครงสร้างชั้น Pooling ไปจนถึง Explainable AI"
        if is_th else
        "Audited workflow from cryptographic deduplication to two-stage transfer learning and post-hoc visual attribution."
    )

    if is_th:
        steps = [
            ("1. การรวบรวมและจัดเตรียมชุดข้อมูล (Dataset Collection & Standardization)",
             "รวบรวมภาพใบข้าวความละเอียดสูงจำนวน **3,829 ภาพ** ครอบคลุม 6 คลาส (5 โรคใบข้าวสำคัญ และ 1 ใบข้าวปกติ) พร้อมปรับขนาดมาตรฐานเป็น `224×224` พิกเซล (RGB)"),
            ("2. การคัดกรองภาพซ้ำ 2 ขั้นตอนและการแบ่งชุดข้อมูลไร้การรั่วไหล (SHA-256 + pHash)",
             "**ขั้นที่ 1 (SHA-256 Exact Dedup):** ตรวจสอบไฟล์ซ้ำ 100% ด้วย Cryptographic Hash และตัดภาพซ้ำออก **31 ภาพ** เหลือภาพไม่ซ้ำ **3,798 ภาพ**\n\n"
             "**ขั้นที่ 2 (pHash Near-Duplicate Grouping):** คำนวณ 64-bit Perceptual Hash (`pHash`, Hamming distance ≤ 10) เพื่อจัดกลุ่มภาพที่คล้ายกันหรือถ่ายต่อเนื่อง **67 กลุ่ม** ให้อยู่ใน Split เดียวกันเสมอ แบ่งเป็น **Train 75% (2,847 ภาพ)**, **Validation 10% (381 ภาพ)** และ **Test 15% (570 ภาพ)** โดยไม่มีภาพรั่วไหลข้ามชุด (Zero Leakage)"),
            ("3. โครงสร้างชั้น Pooling และหัวจำแนกคลาส (Pooling Layer & Classification Head Flow)",
             "**EfficientNetV2-S (Spatial Global Average Pooling):**\n"
             "`Input (224×224×3)` → `Fused-MBConv Backbone (7×7×1280 Feature Map)` → **`GlobalAveragePooling2D (1280-d)`** → `Dense(256, ReLU)` → `Dropout(0.3)` → `Dense(6, Softmax)` — ยุบรวม Feature Map ขนาด 7×7 ของแต่ละ Channel ให้เป็นค่าเฉลี่ยเพียงค่าเดียวแทนการใช้ `Flatten()` ช่วยลดจำนวนพารามิเตอร์ ป้องกัน Overfitting และรักษาตำแหน่งเชิงพื้นที่สำหรับทำ Grad-CAM++\n\n"
             "**ViT-B/16 (CLS Token Pooling & 16×16 Saliency Average Pooling):**\n"
             "`Input (224×224×3)` → `16×16 Patch Projection (196 Patch Tokens + 1 [CLS] Token)` → `12× Transformer Encoder` → **`Token Pooling ([CLS] 768-d)`** → `Dropout(0.1)` → `Dense(6, Softmax)` — สรุปความสัมพันธ์จากทุก Patch ลงใน `[CLS]` Token และในขั้นตอนสร้าง Heatmap จะใช้ **`tf.nn.avg_pool2d (ksize=16, strides=16)`** ยุบ Input Gradient ขนาด `224×224` ลงเป็นตาราง **14×14 Patch Grid** ให้ตรงกับขนาด Patch ของ ViT"),
            ("4. การเทรนและปรับจูนโมเดล 2 ระยะ (Two-Stage Transfer Learning & Fine-Tuning)",
             "**Stage 1 (Freeze Backbone):** ล็อกน้ำหนัก Backbone ที่เทรนจาก ImageNet (`trainable = False`) แล้วเทรนเฉพาะส่วนหัวจำแนกคลาสด้วย `Adam (lr = 1e-3)` และ `EarlyStopping (patience = 3)`\n\n"
             "**Stage 2 (Unfreeze & Full Fine-Tuning):** ปลดล็อกทั้งโครงข่าย (`trainable = True`) แล้วปรับจูนด้วย `AdamW (weight_decay = 0.05)` โดยใช้ Learning Rate ต่ำ (`5e-5` สำหรับ EfficientNetV2-S และ `2e-5` สำหรับ ViT-B/16) ร่วมกับ `ReduceLROnPlateau` และ `EarlyStopping`"),
            ("5. Explainable AI: Pre-Softmax Grad-CAM++ และ 14×14 Attention Saliency",
             "**EfficientNetV2-S (Grad-CAM++):** คำนวณ Gradient จากค่า Pre-softmax Logits ย้อนกลับไปหา Feature Map ชั้นสุดท้าย พร้อมถ่วงน้ำหนักด้วยอนุพันธ์อันดับ 2 และ 3 (Chattopadhay et al., 2018) เพื่อระบุขอบเขตรอยโรคหลายจุดได้อย่างคมชัดโดยไม่เกิดปัญหา Softmax อิ่มตัว\n\n"
             "**ViT-B/16 (Attention Saliency):** คำนวณ Input-Gradient Saliency ($|\\partial \\text{score} / \\partial x|$) ยุบรวมเป็นตาราง 14×14 Patches ตัดค่าสุดโต่งที่ Percentile 99 และซ้อนทับภาพต้นฉบับด้วย `COLORMAP_JET` (`α = 0.45`)"),
        ]
    else:
        steps = [
            ("1. Dataset Collection & Standardization",
             "3,829 high-resolution foliar rice images across six pathological classes standardized to 224×224 RGB tensors."),
            ("2. Two-Stage Deduplication & Zero-Leakage Split (SHA-256 + pHash)",
             "Stage 1 removed **31 exact duplicate files** via SHA-256 cryptographic hashing, preserving **3,798 unique canonical images**. "
             "Stage 2 computed 64-bit DCT perceptual hashes (`pHash`, Hamming distance ≤ 10) to isolate **67 near-duplicate clusters** "
             "(burst shots and pre-augmented variants) into identical splits: **75% Train (2,847)**, **10% Validation (381)**, and **15% Test (570)**."),
            ("3. Pooling Layer & Classification Head Flow",
             "**EfficientNetV2-S (Spatial Global Average Pooling):**\n"
             "`Input (224×224×3)` → `Fused-MBConv Backbone (7×7×1280 Feature Map)` → **`GlobalAveragePooling2D (1280-d)`** → `Dense(256, ReLU)` → `Dropout(0.3)` → `Dense(6, Softmax)`. "
             "Collapses each 7×7 spatial channel into its mean activation rather than using `Flatten()`, preventing parameter bloat and preserving channel attribution for Grad-CAM++.\n\n"
             "**ViT-B/16 (CLS Token Pooling & 16×16 Saliency Average Pooling):**\n"
             "`Input (224×224×3)` → `16×16 Patch Projection (196 Patch Tokens + 1 [CLS] Token)` → `12× Transformer Encoder` → **`Token Pooling ([CLS] 768-d)`** → `Dropout(0.1)` → `Dense(6, Softmax)`. "
             "During XAI attribution, pixel-level gradients (`224×224`) are downsampled via **`tf.nn.avg_pool2d (ksize=16, strides=16)`** into a **14×14 patch grid** aligned with the transformer's native token geometry."),
            ("4. Two-Stage Transfer Learning & Fine-Tuning",
             "**Stage 1 (Head Adaptation):** ImageNet-pretrained backbones frozen (`trainable = False`); classification head trained with `Adam (lr = 1e-3)` and `EarlyStopping (patience = 3)`.\n\n"
             "**Stage 2 (Full Fine-Tuning):** Full network unfrozen with `AdamW (weight_decay = 0.05)` at low learning rates (`5e-5` for EfficientNetV2-S, `2e-5` for ViT-B/16), governed by `ReduceLROnPlateau` and `EarlyStopping`."),
            ("5. Explainable AI: Pre-Softmax Grad-CAM++ & 14×14 Attention Saliency",
             "**EfficientNetV2-S:** Uses pre-softmax logits with 2nd- and 3rd-order positive gradient power weighting (Chattopadhay et al., 2018) to avoid softmax saturation and localize multi-lesion borders.\n\n"
             "**ViT-B/16:** Computes input-gradient saliency ($|\\partial \\text{score} / \\partial x|$) pooled via `avg_pool2d (16×16)` into a 14×14 token grid, clipped at the 99th percentile and blended with `COLORMAP_JET` (`α = 0.45`)."),
        ]

    for title, body in steps:
        with st.container(border=True):
            st.subheader(title)
            st.markdown(body)
