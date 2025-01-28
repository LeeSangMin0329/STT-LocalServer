import pyaudio
import webrtcvad
import time
import numpy as np
from typing import Callable
from queue import Queue
from constants import SAMPLE_RATE

# Audio settings
CHUNK_SIZE = 320  # 320프레임으로 설정 (20ms)
STREAM_DELAY_SEC = CHUNK_SIZE / SAMPLE_RATE

CHANNELS = 1  # 채널 수
SILENCE_LIMIT = 1  # 침묵 제한 (초)
DEVICE_INDEX = 1

recorded_audio = Queue(maxsize=163840) # 104,857,600(100mb) / (chunk size * 2(16bit sample))

def record_stream_from_microphone():
    vad_model = webrtcvad.Vad()
    vad_model.set_mode(3)
    p = pyaudio.PyAudio()

    # 마이크 스트리밍 시작
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=DEVICE_INDEX,
                    frames_per_buffer=CHUNK_SIZE)

    # 음성이 감지된 후 짧은 공백을 유지하기 위한 변수
    silence_counter = 400 # 2 sec
    MAX_SILENCE = 400  # 짧은 침묵 구간을 허용할 최대 횟수

    while True:
        available_frames = stream.get_read_available()

        if available_frames < CHUNK_SIZE:
            time.sleep(STREAM_DELAY_SEC)
            continue

        if recorded_audio.full():
            print("audio steam queue is full")
            continue

        frames = stream.read(CHUNK_SIZE)

        if vad_model.is_speech(frames, SAMPLE_RATE):
            recorded_audio.put(frames)
            silence_counter = 0
        else:
            silence_counter += 1
            if silence_counter <= MAX_SILENCE:
                recorded_audio.put(frames)

def speech_recognition(on_audio_output: Callable[[np.ndarray], None]):
    vad_model = webrtcvad.Vad()
    vad_model.set_mode(3)

    buffer = b""  # 음성 버퍼

    in_speech = False
    silence_threshold = 0
    
    while True:
        if not recorded_audio.empty():
            frames = recorded_audio.get()  # 큐에서 음성 데이터 가져오기

            # VAD를 사용하여 음성 구간만 추출
            is_speech = vad_model.is_speech(frames, sample_rate=SAMPLE_RATE)

            if is_speech:
                if not in_speech:
                    in_speech = True  # 처음 음성이 시작되었을 때

                buffer += frames  # 음성을 버퍼에 추가
                silence_threshold = 0

            elif not is_speech and in_speech:
                if silence_threshold < SILENCE_LIMIT * (SAMPLE_RATE / CHUNK_SIZE):
                    silence_threshold += 1
                else:
                    input_audio = np.frombuffer(buffer, dtype=np.int16)

                    on_audio_output(input_audio) 

                    # 초기화
                    in_speech = False
                    silence_threshold = 0
                    buffer = b""  # 버퍼 초기화
