#!/bin/bash
# ── RiceVision AI — Entrypoint ────────────────────────────────────────────────
# Downloads models from HuggingFace Hub on first boot, then starts FastAPI.
set -e

MODEL_DIR="${MODELS_DIR:-/app/models}"
mkdir -p "$MODEL_DIR"

# ── Download models if not already cached ─────────────────────────────────────
python - <<'EOF'
import os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

# Set your HuggingFace model repo here
HF_REPO = os.getenv("HF_MODEL_REPO", "YOUR_HF_USERNAME/rice-vision-models")
MODEL_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))

models = ["efficientnetv2s.keras", "convnextbase.keras"]

for model_file in models:
    dest = MODEL_DIR / model_file
    if dest.exists():
        print(f"[cache] {model_file} already present, skipping download.", flush=True)
    else:
        print(f"[download] Downloading {model_file} from {HF_REPO} ...", flush=True)
        hf_hub_download(
            repo_id=HF_REPO,
            filename=model_file,
            local_dir=str(MODEL_DIR),
            repo_type="model",
        )
        print(f"[download] Done: {dest}", flush=True)

print("[startup] All models ready.", flush=True)
EOF

# ── Start FastAPI backend ─────────────────────────────────────────────────────
echo "[startup] Starting uvicorn on port 7860 ..."
exec uvicorn backend.main:app --host 0.0.0.0 --port 7860
