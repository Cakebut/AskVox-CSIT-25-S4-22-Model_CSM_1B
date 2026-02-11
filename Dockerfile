# -------------------------------
# Base image with CUDA + PyTorch
# -------------------------------
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# -------------------------------
# System dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    git \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# -------------------------------
# Copy model + prompts into container
# -------------------------------
COPY csm-1b/ ./csm-1b/

# -------------------------------
# Copy application code + requirements
# -------------------------------
COPY app.py .

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    torch \
    transformers \
    soundfile

# -------------------------------
# Environment variables
# -------------------------------
ENV MODEL_DIR=./csm-1b

# -------------------------------
# Start FastAPI server
# -------------------------------
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000"]
