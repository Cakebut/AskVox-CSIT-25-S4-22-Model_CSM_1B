import os
import io
import base64
import torch
import runpod
import soundfile as sf
from transformers import AutoProcessor, CsmForConditionalGeneration

# -------------------------------
# Settings
# -------------------------------
MODEL_ID = os.getenv("MODEL_ID", "cakebut/askvoxcsm-1b")
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Preload model on worker start
# -------------------------------
print(f"[Worker] Loading model from: {MODEL_ID}")
print(f"[Worker] Using device: {device}")

try:
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = CsmForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("[Worker] Model loaded successfully!")
except Exception as e:
    print("[Worker] Error loading model:", e)
    raise e

print("[Worker] Ready to accept requests!")

# -------------------------------
# Generate audio function
# -------------------------------
def generate_audio(text: str) -> bytes:
    # Prefix text with speaker id 0
    text = f"[0]{text}"

    # Tokenize input
    inputs = processor(text, add_special_tokens=True).to(device)

    # Generate audio tensor
    with torch.no_grad():
        audio_tensor = model.generate(
            **inputs,
            output_audio=True
        )

    # Convert to numpy waveform
    waveform = audio_tensor[0].cpu().numpy()

    # Save waveform to in-memory WAV buffer
    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=24000, format="WAV")
    buffer.seek(0)
    return buffer.read()

# -------------------------------
# RunPod serverless handler
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
        print("[Handler] Error:", str(e))
        return {"error": str(e)}

# -------------------------------
# Start RunPod serverless worker
# -------------------------------
runpod.serverless.start({"handler": handler})
