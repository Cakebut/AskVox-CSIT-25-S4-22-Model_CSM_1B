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
# Environment
# -------------------------------
ENV MODEL_ID=cakebut/askvoxcsm-1b
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface

# Create cache directory
RUN mkdir -p /cache/huggingface

# -------------------------------
# Start app
# -------------------------------
CMD ["python", "app.py"]
