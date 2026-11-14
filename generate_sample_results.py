"""
generate_sample_results.py
──────────────────────────
Runs both models on the 6 class sample images and saves pre-computed results
to outputs/sample_results.json — so the deployed app can show demo results
from both models without loading ConvNeXtBase at runtime.

Run once locally before deploying:
    python generate_sample_results.py
"""
import base64, io, json, time
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR   = PROJECT_ROOT / "models"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"

CLASS_NAMES = [
    "Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf",
    "Leaf Blast", "Leaf Scald", "Sheath Blight",
]
DISCLAIMER = (
    "Predictions and Grad-CAM heatmaps are deep learning research outputs intended for decision-support. "
    "Grad-CAM highlights key lesion features influencing classification. Results should be verified with field agronomists."
)


# ── Grad-CAM (copied from main.py) ───────────────────────────────────────────

def _compute_gradcam(img_tensor, model, pred_index=None):
    backbone    = model.layers[1]
    head_layers = model.layers[2:]
    with tf.GradientTape() as tape:
        features = backbone(img_tensor, training=False)
        tape.watch(features)
        x = features
        for layer in head_layers:
            x = layer(x, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(x[0]))
        loss = x[:, pred_index]
    grads       = tape.gradient(loss, features)
    pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap     = features[0] @ pooled[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap)
    heatmap     = tf.maximum(heatmap, 0.0)
    mx          = tf.math.reduce_max(heatmap)
    return (heatmap / mx).numpy() if mx > 0 else heatmap.numpy()


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


def _infer(img: Image.Image, model, model_name: str) -> dict:
    img_rgb     = img.convert("RGB").resize((224, 224))
    arr         = np.array(img_rgb, dtype=np.float32)
    tensor      = tf.convert_to_tensor(np.expand_dims(arr, 0), dtype=tf.float32)

    t0    = time.perf_counter()
    preds = model.predict(tensor, verbose=0)
    ms    = round((time.perf_counter() - t0) * 1000, 1)

    scores    = preds[0].tolist()
    top_idx   = int(np.argmax(scores))
    heatmap   = _compute_gradcam(tensor, model, top_idx)
    overlay_b64, raw_b64 = _overlay(img, heatmap)

    return {
        "model":           model_name,
        "model_label":     "EfficientNetV2-S" if "eff" in model_name else "ConvNeXtBase",
        "predicted_class": CLASS_NAMES[top_idx],
        "predicted_index": top_idx,
        "confidence":      round(float(scores[top_idx]), 4),
        "scores":          {CLASS_NAMES[i]: round(float(scores[i]), 4) for i in range(len(CLASS_NAMES))},
        "inference_ms":    ms,
        "gradcam_overlay": overlay_b64,
        "gradcam_raw":     raw_b64,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    class_images_path = OUTPUTS_DIR / "class_images.json"
    if not class_images_path.exists():
        raise FileNotFoundError(f"Run the backend first to generate {class_images_path}")

    with open(class_images_path, encoding="utf-8") as f:
        class_images: dict = json.load(f)

    print("Loading EfficientNetV2-S …")
    eff_model  = tf.keras.models.load_model(str(MODELS_DIR / "efficientnetv2s.keras"))
    print("Loading ConvNeXtBase …")
    conv_model = tf.keras.models.load_model(str(MODELS_DIR / "convnextbase.keras"))

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

        eff_res  = _infer(img, eff_model,  "efficientnetv2s")
        conv_res = _infer(img, conv_model, "convnextbase")

        results[cls_name] = {
            "mode":            "both",
            "efficientnetv2s": eff_res,
            "convnextbase":    conv_res,
            "disclaimer":      DISCLAIMER,
        }
        print(f"  EfficientNet → {eff_res['predicted_class']} ({eff_res['confidence']:.2%})")
        print(f"  ConvNeXt     → {conv_res['predicted_class']} ({conv_res['confidence']:.2%})")

    out_path = OUTPUTS_DIR / "sample_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"\n✅ Saved → {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
