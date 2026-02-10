# Base image with CUDA + PyTorch
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Copy app and requirements
COPY app.py .
COPY requirements.txt .
COPY prompts/ prompts/

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables
ENV MODEL_ID=cakebut/askvoxcsm-1b
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

# Run the serverless app
CMD ["python", "app.py"]
