import os
import io
import base64
import torch
import soundfile as sf
import runpod

from transformers import AutoProcessor, CsmForConditionalGeneration
from huggingface_hub import hf_hub_download

# =====================
# Configuration
# =====================
REPO_ID = "cakebut/askvoxcsm-1b"
SUBFOLDER = "csm-1b"
HF_TOKEN = os.getenv("HF_TOKEN", None)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CSM model from {REPO_ID}/{SUBFOLDER} on {DEVICE}...")

# =====================
# Load processor & model
# =====================
processor = AutoProcessor.from_pretrained(
    REPO_ID,
    subfolder=SUBFOLDER,
    trust_remote_code=True,
    token=HF_TOKEN
)

model = CsmForConditionalGeneration.from_pretrained(
    REPO_ID,
    subfolder=SUBFOLDER,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    token=HF_TOKEN
)

model.eval()
print("Model loaded successfully!")

# =====================
# Voice prompts (download from HF)
# =====================
VOICE_FILES = {
    "conversational_a": "prompts/conversational_a.wav",
    "conversational_b": "prompts/conversational_b.wav",
    "read_a": "prompts/read_speech_a.wav",
    "read_b": "prompts/read_speech_b.wav",
    "read_c": "prompts/read_speech_c.wav",
    "read_d": "prompts/read_speech_d.wav",
}

DEFAULT_VOICE = "conversational_a"
PROMPT_AUDIO = {}

print("Downloading voice prompts...")

for voice, file_path in VOICE_FILES.items():
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=file_path,
            subfolder=SUBFOLDER,
            token=HF_TOKEN
        )
        audio, sr = sf.read(local_path)
        PROMPT_AUDIO[voice] = (audio, sr)
        print(f"Loaded prompt: {voice}")
    except Exception as e:
        print(f"Failed to load {voice}: {e}")

if not PROMPT_AUDIO:
    raise RuntimeError("No prompts loaded. Worker cannot start.")

print("All prompts ready!")

# =====================
# Audio generation
# =====================
def generate_audio(text: str, voice: str) -> bytes:
    if voice not in PROMPT_AUDIO:
        voice = DEFAULT_VOICE

    audio_prompt, sr = PROMPT_AUDIO[voice]

    # CSM expects speaker prefix
    text_input = f"[0]{text}"

    # Tokenize
    inputs = processor(
        text_input,
        add_special_tokens=True,
        return_tensors="pt"
    ).to(DEVICE)

    # Generate audio tokens
    with torch.no_grad():
        audio_tokens = model.generate(
            **inputs,
            max_new_tokens=1024,
            output_audio=True
        )

    # Decode to waveform
    waveform = processor.decode(audio_tokens[0])

    # Convert to WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=24000, format="WAV")
    buffer.seek(0)

    return buffer.read()


# =====================
# RunPod handler
# =====================
def handler(job):
    job_input = job.get("input", {})

    text = job_input.get("text")
    voice = job_input.get("voice", DEFAULT_VOICE)

    if not text:
        return {"error": "Missing input.text"}

    try:
        audio_bytes = generate_audio(text, voice)
        audio_b64 = base64.b64encode(audio_bytes).decode()

        return {
            "audio_base64": audio_b64,
            "format": "wav",
            "sample_rate": 24000,
            "voice_used": voice
        }

    except Exception as e:
        print("Generation error:", str(e))
        return {"error": str(e)}


# =====================
# Start RunPod serverless
# =====================
print("Worker ready. Waiting for jobs...")
runpod.serverless.start({"handler": handler})
