import os
import io
import base64
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, CsmForConditionalGeneration

# -------------------------------
# CONFIG
# -------------------------------
REPO_LOCAL_PATH = "./csm-1b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_VOICE = "conversational_a"

# -------------------------------
# Load processor & model
# -------------------------------
processor = AutoProcessor.from_pretrained(REPO_LOCAL_PATH, trust_remote_code=True)
model = CsmForConditionalGeneration.from_pretrained(
    REPO_LOCAL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# -------------------------------
# Load voice prompts
# -------------------------------
PROMPT_AUDIO = {}
PROMPT_FILES = {
    "conversational_a": "prompts/conversational_a.wav",
    "conversational_b": "prompts/conversational_b.wav",
    "read_a": "prompts/read_speech_a.wav",
    "read_b": "prompts/read_speech_b.wav",
    "read_c": "prompts/read_speech_c.wav",
    "read_d": "prompts/read_speech_d.wav",
}

for voice, path in PROMPT_FILES.items():
    local_path = os.path.join(REPO_LOCAL_PATH, path)
    if os.path.exists(local_path):
        audio, sr = sf.read(local_path)
        PROMPT_AUDIO[voice] = (audio, sr)

if not PROMPT_AUDIO:
    raise RuntimeError("No prompts loaded. Worker cannot start.")

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="AskVox CSM-1B Server")

class GenerateRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE

@app.get("/ping")
def ping():
    return {"status": "ok", "message": "Model is ready"}

@app.post("/generate")
def generate(req: GenerateRequest):
    voice = req.voice if req.voice in PROMPT_AUDIO else DEFAULT_VOICE
    audio_prompt, sr = PROMPT_AUDIO[voice]

    # CSM expects speaker prefix
    text_input = f"[0]{req.text}"

    # Tokenize
    inputs = processor(text_input, return_tensors="pt").to(DEVICE)

    # Generate audio tokens
    with torch.no_grad():
        audio_tokens = model.generate(**inputs, max_new_tokens=1024, output_audio=True)

    # Decode waveform
    waveform = processor.decode(audio_tokens[0])

    # Convert to WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=24000, format="WAV")
    buffer.seek(0)
    audio_b64 = base64.b64encode(buffer.read()).decode()

    return {"audio_base64": audio_b64, "format": "wav", "sample_rate": 24000, "voice_used": voice}
