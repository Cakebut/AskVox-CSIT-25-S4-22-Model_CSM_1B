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

print(f"Loading model from: {MODEL_ID}")
print(f"Using device: {device}")

# -------------------------------
# Load processor + model
# -------------------------------
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = CsmForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",  # automatically use GPU if available
    trust_remote_code=True
)
model.eval()
print("Model loaded successfully!")

# -------------------------------
# Generate audio
# -------------------------------
def generate_audio(text: str) -> bytes:
    # Use default speaker 0
    text = f"[0]{text}"

    # Tokenize input
    inputs = processor(text, add_special_tokens=True).to(device)

    # Generate audio tensor
    with torch.no_grad():
        audio_tensor = model.generate(
            **inputs,
            output_audio=True,
            max_new_tokens=15000,   
            do_sample=True,
            temperature=0.7
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
        print("Error:", str(e))
        return {"error": str(e)}

# -------------------------------
# Start RunPod worker
# -------------------------------
print("Worker ready.")
runpod.serverless.start({"handler": handler})
