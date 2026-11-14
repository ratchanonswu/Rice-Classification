"""
upload_models_to_hf.py
──────────────────────
Uploads the two Keras model files to a HuggingFace Hub model repository.
Run this ONCE from your local machine before deploying.

Usage:
    pip install huggingface_hub
    huggingface-cli login          # enter your HF token
    python upload_models_to_hf.py
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ── CHANGE THESE ──────────────────────────────────────────────────────────────
HF_USERNAME = "YOUR_HF_USERNAME"           # your HuggingFace username
REPO_NAME   = "rice-vision-models"         # repo will be created if it doesn't exist
# ─────────────────────────────────────────────────────────────────────────────

REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"
MODELS_DIR  = Path(__file__).resolve().parent / "models"

MODEL_FILES = [
    "efficientnetv2s.keras",   # ~83 MB
    "convnextbase.keras",      # ~338 MB
]

def main():
    api = HfApi()

    # Create the repo if it doesn't exist (private=False → public)
    print(f"Creating/verifying repo: {REPO_ID}")
    create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)

    for filename in MODEL_FILES:
        local_path = MODELS_DIR / filename
        if not local_path.exists():
            print(f"[skip] {filename} not found at {local_path}")
            continue

        size_mb = round(local_path.stat().st_size / 1024 / 1024, 1)
        print(f"[upload] {filename} ({size_mb} MB) → {REPO_ID} ...")

        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"[upload] Done: https://huggingface.co/{REPO_ID}/blob/main/{filename}")

    print("\n✅ All models uploaded!")
    print(f"   Set HF_MODEL_REPO={REPO_ID} in your HuggingFace Space secrets.")

if __name__ == "__main__":
    main()
