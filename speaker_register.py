# speaker_register.py

import librosa
import tempfile
import os

from fastapi import APIRouter, UploadFile, File, Form
from speaker_service import extract_embedding, register_speaker, identify_speaker

router = APIRouter()

@router.post("/register-speaker")
async def register_speaker_api(
    full_name: str = Form(...),
    file: UploadFile = File(...)
):
    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    audio_array, sr = librosa.load(tmp_path, sr=16000)

    embedding = extract_embedding(audio_array)

    register_speaker(full_name, embedding)

    os.remove(tmp_path)

    return {
        "status": "registered",
        "name": full_name
    }

# 🔥 YANGA QO'SHILGAN TEST API
@router.post("/test-speaker")
async def test_speaker_api(
    file: UploadFile = File(...)
):
    """
    Bu API faqat bitta audio faylni qabul qilib, uning ichidagi 
    asosiy ovoz egasini va aniqlik foizini (confidence) qaytaradi.
    """
    audio_bytes = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Audioni yuklash
    audio_array, sr = librosa.load(tmp_path, sr=16000)

    # Ovozni aniqlash
    speaker_name, confidence = identify_speaker(audio_array)

    os.remove(tmp_path)

    return {
        "status": "tested",
        "predicted_name": speaker_name,
        "confidence": confidence,
        "note": "Agar ism 'Unknown' bo'lsa, demak confidence 0.75 dan past bo'lgan."
    }
