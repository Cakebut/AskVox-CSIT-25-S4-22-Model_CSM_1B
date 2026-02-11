import os
import io
import base64
import torch
import runpod

from transformers import AutoProcessor, CsmForConditionalGeneration
import soundfile as sf

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
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = CsmForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True
)

model.eval()
print("Model loaded successfully!")

# -------------------------------
# Generate audio
# -------------------------------
def generate_audio(text, speaker_id="0"):
    # Prepare conversation for single sentence
    conversation = [
        {"role": speaker_id, "content": [{"type": "text", "text": text}]}
    ]
    
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True
    ).to(device)

    with torch.no_grad():
        audio = model.generate(**inputs, output_audio=True)

    # Convert to WAV in memory
    buffer = io.BytesIO()
    sf.write(buffer, audio[0].cpu().numpy(), samplerate=24000, format="WAV")
    buffer.seek(0)
    return buffer.read()

# -------------------------------
# RunPod handler
# -------------------------------
def handler(job):
    job_input = job.get("input", {})
    text = job_input.get("text")
    speaker_id = str(job_input.get("speaker_id", "0"))

    if not text:
        return {"error": "Missing input.text"}

    try:
        audio_bytes = generate_audio(text, speaker_id)
        audio_b64 = base64.b64encode(audio_bytes).decode()

        return {
            "audio_base64": audio_b64,
            "sample_rate": 24000,
            "format": "wav"
        }

    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}

# -------------------------------
# Start worker
# -------------------------------
print("Worker ready.")
runpod.serverless.start({"handler": handler})
