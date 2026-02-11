import os
import torch
from transformers import AutoModel, AutoProcessor

# -------------------------------
# Settings
# -------------------------------
MODEL_ID = os.getenv("MODEL_ID", "cakebut/askvoxcsm-1b")

print("Loading model from:", MODEL_ID)

# -------------------------------
# Device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------
# Load processor + model
# -------------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

model.eval()

print("Model loaded successfully!")

# -------------------------------
# Keep container alive (RunPod)
# -------------------------------
while True:
    pass
