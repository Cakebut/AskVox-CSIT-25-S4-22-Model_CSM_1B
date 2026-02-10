FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps (audio)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Python deps (minimal to avoid large image)
RUN pip install \
    runpod \
    transformers \
    accelerate \
    soundfile \
    sentencepiece \
    huggingface_hub

# Set HF cache inside image
ENV HF_HOME=/app/hf_cache

# ---- Download model at build time ----
# IMPORTANT: replace with your HF repo
ENV MODEL_ID=cakebut/askvoxcsm-1b

RUN python -c "from transformers import AutoProcessor, AutoModel; \
AutoProcessor.from_pretrained('$MODEL_ID', trust_remote_code=True); \
AutoModel.from_pretrained('$MODEL_ID', trust_remote_code=True)"

# Copy app
COPY app.py .

CMD ["python", "app.py"]
