# -------------------------------
# Base image
# -------------------------------
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# -------------------------------
# System dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Python dependencies first (better caching)
# -------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Copy app after deps
# -------------------------------
COPY app.py .

# -------------------------------
# Environment variables
# -------------------------------
ENV MODEL_ID=cakebut/askvoxcsm-1b
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface

RUN mkdir -p /cache/huggingface

# -------------------------------
# Start worker
# -------------------------------
CMD ["python", "app.py"]
