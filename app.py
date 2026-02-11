import os
import io
import base64
import torch
import runpod

from transformers import AutoProcessor, CsmForConditionalGeneration

# -------------------------------
# Settings
# -------------------------------
MODEL_ID = os.getenv("MODEL_ID", "cakebut/askvoxcsm-1b")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from: {MODEL_ID}")
print(f"Using device: {device}")

# -------------------------------
# Load processor + model
# -------------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = CsmForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map=device,
    trust_remote_code=True
)
model.eval()
print("Model loaded successfully!")

# -------------------------------
# Generate audio
# -------------------------------
def generate_audio(text: str) -> bytes:
    # CSM expects a speaker prefix
    text = f"[0]{text}"

    # Tokenize input
    inputs = processor(text, add_special_tokens=True).to(device)

    # Generate audio
    with torch.no_grad():
        audio = model.generate(**inputs, output_audio=True)

    # Save audio to buffer
    buffer = io.BytesIO()
    processor.save_audio(audio, buffer)
    buffer.seek(0)
    return buffer.read()

# -------------------------------
# RunPod handler
# -------------------------------
def handler(job):
    text = job.get("input", {}).get("text")
    if not text:
        return {"error": "Missing input.text"}

    try:
        audio_bytes = generate_audio(text)
        audio_b64 = base64.b64encode(audio_bytes).decode()
        return {
            "audio_base64": audio_b64,
            "format": "wav",
            "sample_rate": 24000
        }
    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# Start RunPod worker
# -------------------------------
print("Worker ready.")
runpod.serverless.start({"handler": handler})
