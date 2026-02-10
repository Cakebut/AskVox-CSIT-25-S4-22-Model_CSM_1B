import base64
import io
import runpod
import torch
import soundfile as sf
from transformers import AutoProcessor, AutoModel

# -----------------------
# Config
# -----------------------
MODEL_ID = "cakebut/askvoxcsm-1b"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Loading CSM-1B from local cache...")

# Load from cache (downloaded during Docker build)
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    trust_remote_code=True
).to(device)

model.eval()

print("CSM-1B loaded successfully on", device)

# -----------------------
# TTS generation
# -----------------------
def generate_audio(text: str):
    inputs = processor(text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        audio = model.generate(**inputs)

    audio = audio.cpu().numpy().squeeze()

    # Save to memory as WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate=22050, format="WAV")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


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
