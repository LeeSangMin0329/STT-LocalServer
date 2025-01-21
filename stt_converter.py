import torch
import time
import numpy as np
from faster_whisper import WhisperModel
from preprocess_audio import preprocessing
from postprocess_text import clean_iterable


# whisper settings
MODEL_SIZE = "medium"
DEVICE = "cuda"

whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="float16")

def convert_to_text(audio: np.ndarray) -> str:
    start = time.time()
    print(f"STT start {start}") 

    # preprocessing
    preprocessed_audio = preprocessing(audio)

    preprocessing_time = time.time()
    print(f"preprocessing_time delay : {preprocessing_time - start} seconds.")

    segments, _ = whisper_model.transcribe(preprocessed_audio, language='ja', beam_size=10, vad_filter=True, without_timestamps=True,)

    # postprocessing
    text_result = clean_iterable(segments)

    end = time.time()
    length = end - start

    print(f"Convert delay : {length} seconds. {len(list(segments))}")
    print(f"Detected: {text_result}")
    return text_result
