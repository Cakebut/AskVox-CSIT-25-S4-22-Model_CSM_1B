# -------------------------------
# Base image
# -------------------------------
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# -------------------------------
# Environment variables
# -------------------------------
ENV MODEL_ID=cakebut/askvoxcsm-1b
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# -------------------------------
# System dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create cache directory
RUN mkdir -p /cache/huggingface

# -------------------------------
# Install Python dependencies
# -------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Pre-download model (IMPORTANT)
# -------------------------------
RUN python -c "from transformers import AutoProcessor, CsmForConditionalGeneration; \
model_id='cakebut/askvoxcsm-1b'; \
AutoProcessor.from_pretrained(model_id, trust_remote_code=True); \
CsmForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True); \
print('Model downloaded successfully')"

# -------------------------------
# Copy application
# -------------------------------
COPY app.py .

# -------------------------------
# Start worker
# -------------------------------
CMD ["python", "-u", "app.py"]
