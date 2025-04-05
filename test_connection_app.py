import io
import base64
import requests
import numpy as np
import soundfile as sf
from constants import SAMPLE_RATE

from preprocess_audio import is_invalid_audio

server_url = "http://127.0.0.1:50003"

def test_request_play(message: str):
    if message.strip() == "":
        print("is empty string")
        return 

    try:
        response = requests.post(f"{server_url}/chat", data = {"message": message, "required_translate" : "False"})

        print(f"Send to {server_url} : {response.status_code}")
    except Exception as e:
        print(f"Exception : {str(e)}")

def request_audio(audio: np.ndarray):
    if is_invalid_audio(audio, SAMPLE_RATE):
        print("invalid audio detected")
        return
    
    audio_buffer = io.BytesIO()

    sf.write(audio_buffer, audio, SAMPLE_RATE, format="WAV")
    audio_buffer.seek(0)

    # encoded_data = base64.b64encode(audio_buffer.read()).decode("utf-8")

    try:
        response = requests.post(f"{server_url}/chat_audio", files = {"message" : ("audio_data.wav", audio_buffer, "audio/wav")})

        print(f"Send to {server_url} : {response.status_code}")
    except Exception as e:
        print(f"Exception : {str(e)}")
