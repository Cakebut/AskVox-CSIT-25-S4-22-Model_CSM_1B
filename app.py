import os
import io
import base64
import torch
import soundfile as sf
import runpod

from transformers import AutoProcessor, CsmForConditionalGeneration
from huggingface_hub import hf_hub_download, snapshot_download

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model folder (already downloaded in Dockerfile)
MODEL_PATH = "./csm-1b"

print(f"Loading CSM model from {MODEL_PATH} on {DEVICE}...")

# Load processor and model
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = CsmForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

model.eval()
print("✅ Model loaded successfully!")

# Load prompts
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
    try:
        local_path = os.path.join(MODEL_PATH, path)
        audio, sr = sf.read(local_path)
        PROMPT_AUDIO[voice] = (audio, sr)
        print(f"Loaded prompt: {voice}")
    except Exception as e:
        print(f"Failed to load {voice}: {e}")

DEFAULT_VOICE = "conversational_a"
if not PROMPT_AUDIO:
    raise RuntimeError("No prompts loaded!")

# Audio generation
def generate_audio(text: str, voice: str) -> bytes:
    if voice not in PROMPT_AUDIO:
        voice = DEFAULT_VOICE
    audio_prompt, sr = PROMPT_AUDIO[voice]

    text_input = f"[0]{text}"

    inputs = processor(
        text_input,
        add_special_tokens=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        audio_tokens = model.generate(
            **inputs,
            max_new_tokens=1024,
            output_audio=True
        )

    waveform = processor.decode(audio_tokens[0])

    buffer = io.BytesIO()
    sf.write(buffer, waveform, samplerate=24000, format="WAV")
    buffer.seek(0)

    return buffer.read()

# RunPod handler
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

print("Worker ready. Waiting for jobs...")
runpod.serverless.start({"handler": handler})
