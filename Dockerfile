# ── RiceVision AI — HuggingFace Spaces Docker Image ──────────────────────────
# Base image: slim Python 3.11
FROM python:3.11-slim

# HuggingFace Spaces requires the app to run as a non-root user (UID 1000)
RUN useradd -m -u 1000 user
WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
# libgl1 and libglib2.0-0 are needed by OpenCV headless at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY RiceVisionAI/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── App source ────────────────────────────────────────────────────────────────
COPY RiceVisionAI/backend/ ./backend/
COPY RiceVisionAI/frontend/dist/ ./frontend/dist/
COPY outputs/ ./outputs/

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# Give the non-root user ownership
RUN chown -R user:user /app
USER user

# Models will be downloaded to /app/models at startup
ENV MODELS_DIR=/app/models
ENV OUTPUTS_DIR=/app/outputs
ENV FRONTEND_DIST=/app/frontend/dist

# HuggingFace Spaces exposes port 7860
EXPOSE 7860

CMD ["./entrypoint.sh"]
