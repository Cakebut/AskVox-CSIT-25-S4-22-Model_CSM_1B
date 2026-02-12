# -------------------------------
# Base image (smaller PyTorch + CUDA runtime)
# -------------------------------
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn11-runtime

WORKDIR /app

# -------------------------------
# System dependencies
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Copy project files
# -------------------------------
COPY app.py .
COPY requirements.txt .

# -------------------------------
# Install Python dependencies
# -------------------------------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Environment variables
# -------------------------------
ENV MODEL_ID=cakebut/askvoxcsm-1b
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface

# Create cache directory (huggingface model cache)
RUN mkdir -p /cache/huggingface

# -------------------------------
# RunPod worker entrypoint
# -------------------------------
CMD ["python", "app.py"]
