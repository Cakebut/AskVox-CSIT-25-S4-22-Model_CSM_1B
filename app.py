import os
import io
import base64
import torch
import soundfile as sf
import runpod
from transformers import AutoProcessor, CsmForConditionalGeneration

# -----------------------
# Configuration
# -----------------------
MODEL_ID = os.getenv("MODEL_ID", "cakebut/askvoxcsm-1b")
HF_TOKEN = os.getenv("HF_TOKEN", None)  # required if repo is private
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CSM-1B model from {MODEL_ID} on {DEVICE}...")

# Load processor & model
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN
)
model = CsmForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=HF_TOKEN
)
model.eval()
print("Model loaded successfully!")

# -----------------------
# TTS generation
# -----------------------
def generate_audio(text: str) -> str:
    # Encode text
    inputs = processor(text=text, return_tensors="pt").to(DEVICE)

    # Generate audio tokens
    with torch.no_grad():
        audio_tokens = model.generate(**inputs)

    # Decode audio tokens to waveform
    audio_waveform = processor.decode(audio_tokens[0])

    # Save waveform to buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio_waveform, samplerate=24000, format="WAV")
    buffer.seek(0)

    # Return base64 string
    return base64.b64encode(buffer.read()).decode()

# -----------------------
# RunPod serverless handler
# -----------------------
def handler(job):
    inp = job.get("input", {})
    text = inp.get("text")

    if not text:
        return {"error": "Missing input.text"}
    if not isinstance(text, str):
        return {"error": "input.text must be a string"}

    try:
        audio_b64 = generate_audio(text)
        return {
            "audio_base64": audio_b64,
            "format": "wav",
            "sample_rate": 24000
        }
    except Exception as e:
        return {"error": str(e)}

# -----------------------
# Start serverless
# -----------------------
runpod.serverless.start({"handler": handler})
