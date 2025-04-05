import numpy as np
import noisereduce
import librosa
from scipy.signal import butter, lfilter
from constants import SAMPLE_RATE

def preprocessing(data):
    denoised = denoise(data)
    trimmed, _ = trim(denoised)
    preemphasised = preemphasis(trimmed)
    moveing_average = moving_average_filter(preemphasised)
    bandpass = bandpass_filter(moveing_average)
    amplified = amplify(bandpass, factor=2.0)
    return normalize_16bits(amplified)

def denoise(data):
    return noisereduce.reduce_noise(data, SAMPLE_RATE)

def trim(data):
    return librosa.effects.trim(data, top_db=20)

def preemphasis(signal, coeff = 0.97):
    """
    Pre-emphasis 필터를 적용하여 고 주파수 대역 강화.
    
    Parameters:
    - signal: 입력 음성 신호 (1D numpy array)
    - coeff: 필터 계수
    
    Returns:
    - 강조된 음성 신호 (1D numpy array)
    """
    emphasized_signal = np.append(signal[0], signal[1:] - coeff * signal[:-1])
    return emphasized_signal

def moving_average_filter(input_audio, window_size = 5):
    """
    이동 평균 필터를 적용하여 갑작스러운 스파이크나 급격한 변화 제거.
    
    Parameters:
    - input_audio: 입력 오디오 (1D numpy array)
    - window_size: 이동 평균을 계산할 윈도우 크기

    Returns:
    - 필터링된 오디오 (1D numpy array)
    """
    return np.convolve(input_audio, np.ones(window_size)/window_size, mode='same')

def bandpass_filter(input_audio):
    """
    bandpass 필터를 적용하여 적용 대역 제한.
    
    Parameters:
    - input_audio: 입력 오디오 (1D numpy array)
    
    Returns:
    - 필터링된 오디오 (1D numpy array)
    """
    lowcut = 300
    highcut = 3400
    order = 5

    nyquist = 0.5 * SAMPLE_RATE
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    return lfilter(b, a, input_audio)

def amplify(input_audio, factor):
    """
    16 비트 오디오 증폭 후 넘는 값 자름.
    
    Parameters:
    - input_audio: 입력 오디오 (1D numpy array)
    - factor: 배율
    
    Returns:
    - 필터링된 오디오 (1D numpy array)
    """
    return np.clip(input_audio * factor, -32768, 32767).astype(np.int16)

def normalize_16bits(input_audio):
    return input_audio.astype(np.float32) / 32768.0  # 16비트 정규화

def is_invalid_audio(audio_data, sample_rate, threshold_db = -25, min_duration = 0.5):
    rms = librosa.feature.rms(y = audio_data)

    db = librosa.amplitude_to_db(rms, ref = np.max)

    is_allow_db = np.any(db > threshold_db)

    duration = librosa.get_duration(y=audio_data, sr=sample_rate)

    is_short = duration <= min_duration
    print(f"is_allow_db {is_allow_db}, duration: {duration}")

    return (not is_allow_db) or is_short