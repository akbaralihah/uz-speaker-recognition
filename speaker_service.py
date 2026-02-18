# speaker_service.py

import torch
import numpy as np
from speechbrain.inference.classifiers import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity

from config import SPEAKER_RECOGNITION_MODEL

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model yuklash
speaker_model = EncoderClassifier.from_hparams(
    source=SPEAKER_RECOGNITION_MODEL,
    run_opts={"device": device}
)

SPEAKER_DB = {}
THRESHOLD = 0.4


def extract_embedding(audio_array):
    with torch.no_grad():
        embedding = speaker_model.encode_batch(
            torch.tensor(audio_array).unsqueeze(0).to(device)
        )

    return embedding.squeeze().cpu().numpy()


def register_speaker(name: str, embedding: np.ndarray):
    SPEAKER_DB[name] = embedding


def identify_speaker(audio_array):
    if not SPEAKER_DB:
        return "Unknown", 0.0

    embedding = extract_embedding(audio_array)

    best_match = None
    best_score = 0

    for name, db_embedding in SPEAKER_DB.items():
        score = cosine_similarity(
            [embedding],
            [db_embedding]
        )[0][0]

        if score > best_score:
            best_score = score
            best_match = name

    if best_score > THRESHOLD:
        return best_match, float(best_score)

    return "Unknown", float(best_score)
