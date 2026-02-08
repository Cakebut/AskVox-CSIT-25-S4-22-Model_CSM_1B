import base64
import io
import runpod
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModel

# -----------------------
# Model loading (cold start)
# -----------------------
MODEL_ID = "YOUR_HF_USERNAME/csm-1b"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CSM-1B model...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
).to(device)

model.eval()

print("CSM-1B loaded successfully")

# -----------------------
# TTS function
# -----------------------
def generate_audio(text: str):
    inputs = processor(text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        audio = model.generate(**inputs)

    audio = audio.cpu().numpy().squeeze()

    # Save to WAV in memory
    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate=22050, format="WAV")
    buffer.seek(0)

    # Convert to base64 for RunPod response
    audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return audio_base64

# -----------------------
# RunPod handler
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
            "sample_rate": 22050
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------
# Start serverless
# -----------------------
runpod.serverless.start({"handler": handler})
