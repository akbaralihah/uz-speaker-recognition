import os
# =============================
# 1️⃣ Environment & Config
# =============================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import logging
import warnings
import torch
import librosa
import tempfile
import asyncio
import time
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import logging as transformers_logging
from pyannote.audio import Pipeline

# 🔥 Speaker system
from speaker_service import identify_speaker
from speaker_register import router as speaker_router
from config import STT_MODEL, DIARIZATION_MODEL

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real-time Speaker Identification API")

# Register speaker API ulash
app.include_router(speaker_router)

# =============================
# 2️⃣ Model Yuklash
# =============================

logger.info(f"Device: {device}")

processor = WhisperProcessor.from_pretrained(STT_MODEL)
model = WhisperForConditionalGeneration.from_pretrained(
    STT_MODEL,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
).to(device)

model.eval()

# Diarization
DIARIZATION_TOKEN = os.getenv("PYANNOTATE_KEY")

diarization_pipeline = Pipeline.from_pretrained(
    DIARIZATION_MODEL,
    token=DIARIZATION_TOKEN
).to(torch.device(device))

logger.info("Barcha modellar yuklandi.")

# =============================
# 3️⃣ WebSocket Endpoint
# =============================

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket ulandi")

    tmp_path = None

    try:
        # 1️⃣ Audio qabul qilish
        data = await websocket.receive_bytes()
        logger.info(f"Audio size: {len(data)/1024/1024:.2f} MB")

        await websocket.send_json({
            "status": "processing",
            "message": "Audio qabul qilindi"
        })

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        start_time = time.perf_counter()

        # 2️⃣ Audio yuklash
        audio_array, sr = librosa.load(tmp_path, sr=16000)

        # 3️⃣ Diarization
        await websocket.send_json({
            "status": "diarizing",
            "message": "Spikerlar aniqlanmoqda..."
        })

        diar_input = {
            "waveform": torch.from_numpy(audio_array).unsqueeze(0),
            "sample_rate": sr
        }

        diar_output = await asyncio.to_thread(
            diarization_pipeline,
            diar_input
        )

        diarization = diar_output.speaker_diarization

        await websocket.send_json({
            "status": "transcribing",
            "message": "Transkripsiya boshlandi..."
        })

        segment_count = 0

        # =============================
        # 4️⃣ Segment Loop
        # =============================

        for turn, _, speaker in diarization.itertracks(yield_label=True):

            await asyncio.sleep(0)

            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)

            segment_audio = audio_array[start_sample:end_sample]

            if len(segment_audio) < 0.5 * sr:
                continue

            # 🔥 4.1 Speaker Identification
            speaker_name, confidence = identify_speaker(segment_audio)

            # 🔥 4.2 STT
            try:
                input_features = processor(
                    segment_audio,
                    sampling_rate=sr,
                    return_tensors="pt"
                ).input_features.to(device)

                if dtype == torch.float16:
                    input_features = input_features.to(dtype)

                with torch.no_grad():
                    predicted_ids = model.generate(
                        input_features,
                        language="uz",
                        task="transcribe",
                        max_new_tokens=256
                    )

                text = processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0].strip()

                if not text:
                    continue

                segment_count += 1

                segment_data = {
                    "type": "segment",
                    "speaker_id": speaker,
                    "speaker_name": speaker_name,
                    "confidence": confidence,
                    "start_ms": int(turn.start * 1000),
                    "end_ms": int(turn.end * 1000),
                    "text": text
                }

                await websocket.send_json(segment_data)

            except Exception as e:
                logger.error(f"Segment error: {e}")
                continue

        total_time = time.perf_counter() - start_time

        await websocket.send_json({
            "type": "finished",
            "total_segments": segment_count,
            "total_time": f"{total_time:.2f}s"
        })

        logger.info(f"Tugadi. Vaqt: {total_time:.2f}s")

    except WebSocketDisconnect:
        logger.info("Client disconnect")

    except Exception as e:
        logger.error(f"Kritik xato: {e}")
        await websocket.send_json({
            "status": "error",
            "message": str(e)
        })

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================
# 4️⃣ Run
# =============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
