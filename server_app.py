import numpy as np
import time
from threading import Thread
from queue import Queue

from audio_stream import record_stream_from_microphone, speech_recognition
from stt_converter import convert_to_text
from test_connection_app import test_request_play, request_audio

text_queue = Queue()

def on_recieved_audio(input_audio: np.ndarray):
    request_audio(input_audio)

    #text_result = convert_to_text(input_audio)
    #text_queue.put(text_result)

def process_audio_data():       
    speech_recognition(on_recieved_audio)

def start_recording():
    print("Starting...")
    record = Thread(target=record_stream_from_microphone)
    record.start()
    transcribe = Thread(target=process_audio_data)
    transcribe.start()
    print("Listening.")

if __name__ == "__main__":
    start_recording()

while True:
    if not text_queue.empty():
        test_request_play(text_queue.get())
    time.sleep(1)
        
     
