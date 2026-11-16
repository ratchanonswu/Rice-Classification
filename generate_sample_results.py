"""
generate_sample_results.py
──────────────────────────
Runs both models (EfficientNetV2-S and ViT-B/16) on the 6 class sample images
and saves pre-computed results to outputs/sample_results.json.

Run once locally before deploying:
    python generate_sample_results.py
"""
import base64, io, json, time
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
import keras_hub
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

CLASS_NAMES = [
    "Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf",
    "Leaf Blast", "Leaf Scald", "Sheath Blight",
]
DISCLAIMER = (
    "Predictions, Grad-CAM, and ViT Saliency heatmaps are deep learning research outputs intended for decision-support. "
    "Heatmaps highlight key lesion features influencing classification. Results should be verified with field agronomists."
)


# ─── Grad-CAM++ (for EfficientNetV2-S) ────────────────────────────────────────

def _compute_gradcam_effnet(img_tensor, model, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
    backbone    = model.layers[1]
    head_layers = model.layers[2:]
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                features = backbone(img_tensor, training=False)
                tape3.watch(features)
                x = features
                for layer in head_layers:
                    x = layer(x, training=False)
                if pred_index is None:
                    pred_index = int(tf.argmax(x[0]))
                score = x[:, pred_index]
            first_grads  = tape3.gradient(score, features)
            second_grads = tape2.gradient(first_grads, features)
            third_grads  = tape1.gradient(second_grads, features)
    alpha_num   = second_grads[0]
    alpha_denom = 2.0 * second_grads[0] + third_grads[0] * features[0] + 1e-10
    alphas      = tf.where(tf.math.is_nan(alpha_num / alpha_denom), tf.zeros_like(alpha_num), alpha_num / alpha_denom)
    weights     = tf.reduce_sum(alphas * tf.maximum(first_grads[0], 0.0), axis=(0, 1))
    cam         = tf.maximum(tf.reduce_sum(weights * features[0], axis=-1), 0.0)
    mx          = tf.math.reduce_max(cam)
    return (cam / mx).numpy() if mx > 0 else cam.numpy()


# ─── ViT Attention Saliency (for ViT-B/16) ────────────────────────────────────

def _compute_vit_saliency(img_tensor, vit_model):
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


def _overlay(pil_img, heatmap, alpha=0.45):
    orig = np.array(pil_img.convert("RGB"))
    h, w, _ = orig.shape
    hm_resized  = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    hm_color    = cv2.applyColorMap(np.uint8(255 * hm_resized), cv2.COLORMAP_JET)
    hm_color    = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    superimposed = np.uint8(orig * (1 - alpha) + hm_color * alpha)

    def to_b64(arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    return to_b64(superimposed), to_b64(hm_color)


def _infer_effnet(img: Image.Image, model) -> dict:
    img_rgb     = img.convert("RGB").resize((224, 224))
    arr         = np.array(img_rgb, dtype=np.float32)
    tensor      = tf.convert_to_tensor(np.expand_dims(arr, 0), dtype=tf.float32)

    t0    = time.perf_counter()
    preds = model.predict(tensor, verbose=0)
    ms    = round((time.perf_counter() - t0) * 1000, 1)

    scores    = preds[0].tolist()
    top_idx   = int(np.argmax(scores))
    heatmap   = _compute_gradcam_effnet(tensor, model, top_idx)
    overlay_b64, raw_b64 = _overlay(img, heatmap)

    return {
        "model":           "efficientnetv2s",
        "model_label":     "EfficientNetV2-S",
        "predicted_class": CLASS_NAMES[top_idx],
        "predicted_index": top_idx,
        "confidence":      round(float(scores[top_idx]), 4),
        "scores":          {CLASS_NAMES[i]: round(float(scores[i]), 4) for i in range(len(CLASS_NAMES))},
        "inference_ms":    ms,
        "gradcam_overlay": overlay_b64,
        "gradcam_raw":     raw_b64,
    }


def _infer_vit(img: Image.Image, model) -> dict:
    img_rgb     = img.convert("RGB").resize((224, 224))
    arr         = (np.array(img_rgb, dtype=np.float32) / 127.5) - 1.0
    tensor      = tf.convert_to_tensor(np.expand_dims(arr, 0), dtype=tf.float32)

    t0    = time.perf_counter()
    preds = model(tensor, training=False).numpy()
    ms    = round((time.perf_counter() - t0) * 1000, 1)

    scores    = preds[0].tolist()
    top_idx   = int(np.argmax(scores))
    heatmap   = _compute_vit_saliency(tensor, model)
    overlay_b64, raw_b64 = _overlay(img, heatmap)

    return {
        "model":           "vit_base",
        "model_label":     "ViT-B/16 (Vision Transformer)",
        "predicted_class": CLASS_NAMES[top_idx],
        "predicted_index": top_idx,
        "confidence":      round(float(scores[top_idx]), 4),
        "scores":          {CLASS_NAMES[i]: round(float(scores[i]), 4) for i in range(len(CLASS_NAMES))},
        "inference_ms":    ms,
        "gradcam_overlay": overlay_b64,
        "gradcam_raw":     raw_b64,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    class_images_path = OUTPUTS_DIR / "class_images.json"
    if not class_images_path.exists():
        raise FileNotFoundError(f"Run the backend first to generate {class_images_path}")

    with open(class_images_path, encoding="utf-8") as f:
        class_images: dict = json.load(f)

    print("Loading EfficientNetV2-S …")
    eff_model = tf.keras.models.load_model(str(MODELS_DIR / "efficientnetv2s.keras"))
    print("Loading ViT-B/16 (vit_base.keras) …")
    vit_model = tf.keras.models.load_model(str(MODELS_DIR / "vit_base.keras"))

    results = {}
    for cls_name in CLASS_NAMES:
        b64 = class_images.get(cls_name)
        if not b64:
            print(f"[skip] No sample image for {cls_name}")
            continue

        print(f"[inference] {cls_name} …")
        # Decode base64 → PIL Image
        img_bytes = base64.b64decode(b64.split(",", 1)[1])
        img = Image.open(io.BytesIO(img_bytes))

        eff_res = _infer_effnet(img, eff_model)
        vit_res = _infer_vit(img, vit_model)

        results[cls_name] = {
            "mode":            "both",
            "efficientnetv2s": eff_res,
            "vit_base":        vit_res,
            "disclaimer":      DISCLAIMER,
        }
        print(f"  EfficientNet → {eff_res['predicted_class']} ({eff_res['confidence']:.2%})")
        print(f"  ViT-B/16     → {vit_res['predicted_class']} ({vit_res['confidence']:.2%})")

    out_path = OUTPUTS_DIR / "sample_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved → {out_path}  ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
