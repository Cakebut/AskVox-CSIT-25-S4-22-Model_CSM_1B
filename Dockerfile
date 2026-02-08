FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy app
COPY app.py .

# Optional: HuggingFace cache location (faster reloads)
ENV HF_HOME=/app/hf_cache

CMD ["python3", "app.py"]
