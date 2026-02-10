# Use NVIDIA PyTorch container
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy files
COPY app.py .
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV RUNPOD_MODEL_ID=sesame/csm-1b

# Expose port (optional, RunPod handles this)
EXPOSE 8080

# Start serverless app
CMD ["python", "app.py"]
