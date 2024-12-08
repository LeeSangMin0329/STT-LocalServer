import torch
import pyaudio
import numpy as np
import librosa
import noisereduce as nr
from scipy.signal import butter, lfilter
import webrtcvad
from queue import Queue
from threading import Thread
from faster_whisper import WhisperModel
import time

model_size = "large-v2"

# faster whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel(model_size, device=device, compute_type="float16")

# vad
# vad_model, utils = torch.hub.load(repo_or_dir='snakers4_silero-vad_master', source="local", model='silero_vad', force_reload=False)
vad_model = webrtcvad.Vad()
vad_model.set_mode(3)

# Audio settings
SAMPLE_RATE = 16000  # 샘플링 레이트
CHUNK_SIZE = 320  # 160프레임으로 설정 (10ms)
STREAM_DELAY_SEC = CHUNK_SIZE / SAMPLE_RATE

AUDIO_FORMAT = pyaudio.paInt16  # 16비트 오디오 포맷
CHANNELS = 1  # 채널 수
SILENCE_LIMIT = 1  # 침묵 제한 (초)
DEVICE_INDEX = 1

# 큐 설정
recordings = Queue()


def preemphasis(signal, coeff=0.97):
    """
    Pre-emphasis 필터를 음성 신호에 적용하는 함수.
    
    Parameters:
    - signal: 입력 음성 신호 (1D numpy array)
    - coeff: 필터 계수 (기본값 0.97)
    
    Returns:
    - 강조된 음성 신호 (1D numpy array)
    """
    emphasized_signal = np.append(signal[0], signal[1:] - coeff * signal[:-1])
    return emphasized_signal

def moving_average_filter(input_audio, window_size=5):
    """
    이동 평균 필터를 적용하여 갑작스러운 스파이크나 급격한 변화 제거.
    
    :param input_audio: 입력 오디오 (numpy array)
    :param window_size: 이동 평균을 계산할 윈도우 크기 (기본값은 5)
    :return: 필터링된 오디오
    """
    return np.convolve(input_audio, np.ones(window_size)/window_size, mode='same')

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def record_microphone():
    p = pyaudio.PyAudio()

    # 마이크 스트리밍 시작
    stream = p.open(format=AUDIO_FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=DEVICE_INDEX,
                    frames_per_buffer=CHUNK_SIZE)

    # 음성이 감지된 후 짧은 공백을 유지하기 위한 변수
    silence_counter = 100
    MAX_SILENCE = 100  # 짧은 침묵 구간을 허용할 최대 횟수

    while True:
        available_frames = stream.get_read_available()

        if available_frames < CHUNK_SIZE:
            time.sleep(STREAM_DELAY_SEC)
            continue

        frames = stream.read(CHUNK_SIZE)

        if vad_model.is_speech(frames, SAMPLE_RATE):
            recordings.put(frames)
            silence_counter = 0
        else:
            silence_counter += 1
            if silence_counter <= MAX_SILENCE:
                recordings.put(frames)

def speech_recognition():
    buffer = b""  # 음성 버퍼
    in_speech = False
    silence_threshold = 0
    
    while True:
        if not recordings.empty():
            frames = recordings.get()  # 큐에서 음성 데이터 가져오기
            # frames = np.frombuffer(frames, dtype=np.int16)  # 바이트 데이터를 int16으로 변환

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

                    start = time.time()
                    print(f"STT start {start}") 
                    # 음성이 끝난 후, 버퍼의 음성을 Whisper에 전달하여 텍스트로 변환
                    input_audio = np.frombuffer(buffer, dtype=np.int16)

                    # preprocessing
                    denoised = nr.reduce_noise(input_audio, SAMPLE_RATE)
                    trimmed, _ = librosa.effects.trim(denoised, top_db=30)
                    preemphasised = preemphasis(trimmed)
                    moveing_average = moving_average_filter(preemphasised)
                    bandpass = bandpass_filter(moveing_average, lowcut=300, highcut=3400, fs=SAMPLE_RATE)

                    normalized_audio = bandpass.astype(np.float32) / 32768.0  # 16비트 정규화
                    segments, _ = whisper_model.transcribe(normalized_audio, language='ko', beam_size=35, temperature=0.6)

                    end = time.time()
                    length = end - start          
                    
                    # 변환된 텍스트 출력
                    for seg in segments:
                        print(f"Detected: {seg.text}")
                    print(f"It took {length} seconds! {len(list(segments))}")          

                    # 초기화
                    in_speech = False
                    silence_threshold = 0
                    buffer = b""  # 버퍼 초기화

def start_recording():
    print("Starting...")
    record = Thread(target=record_microphone)
    record.start()
    transcribe = Thread(target=speech_recognition)
    transcribe.start()
    print("Listening.")

def stop_recording():
    print("Stopped.")
    
if __name__ == "__main__":
    start_recording()
    time.sleep(300)  # 35초 동안 녹음
    stop_recording()