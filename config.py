import os
from dotenv import load_dotenv

load_dotenv()

# modellar
STT_MODEL = os.getenv("STT_MODEL")
DIARIZATION_MODEL=os.getenv("DIARIZATION_MODEL")
SPEAKER_RECOGNITION_MODEL=os.getenv("SPEAKER_RECOGNITION_MODEL")