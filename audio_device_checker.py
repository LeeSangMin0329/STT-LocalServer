import pyaudio
import time

# 오디오 장치 목록 확인
audio = pyaudio.PyAudio()
device_count = audio.get_device_count()

# 장치 목록 출력
for i in range(device_count):
    device_info = audio.get_device_info_by_index(i)
    print(f"Device {i}: {device_info['name']}")


# Audio settings
SAMPLE_RATE = 16000  # 샘플링 레이트
CHUNK_SIZE = 320  # 160프레임으로 설정 (10ms)
STREAM_DELAY_SEC = CHUNK_SIZE / SAMPLE_RATE

AUDIO_FORMAT = pyaudio.paInt16  # 16비트 오디오 포맷
CHANNELS = 1  # 채널 수
SILENCE_LIMIT = 1  # 침묵 제한 (초)
DEVICE_INDEX = 1

stream = audio.open(format=AUDIO_FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=DEVICE_INDEX,
                    frames_per_buffer=CHUNK_SIZE)

stream.start_stream()

try:
    while stream.is_active():
        frames = stream.read(CHUNK_SIZE)
        time.sleep(STREAM_DELAY_SEC) # 10ms

except Exception as e:
    print(f"Callback Error: {e}")