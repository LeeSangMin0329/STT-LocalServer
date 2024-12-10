import numpy as np
import noisereduce as nr
import librosa
from scipy.signal import butter, lfilter
from constants import SAMPLE_RATE

def preprocessing(data):
    denoised = denoise(data)
    trimmed, _ = trim(denoised)
    preemphasised = preemphasis(trimmed)
    moveing_average = moving_average_filter(preemphasised)
    bandpass = bandpass_filter(moveing_average)

    return normalize_16bits(bandpass)

def denoise(data):
    return nr.reduce_noise(data, SAMPLE_RATE)

def trim(data):
    return librosa.effects.trim(data, top_db=20)

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

def bandpass_filter(data):
    lowcut = 300
    highcut = 3400
    order = 5

    nyquist = 0.5 * SAMPLE_RATE
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    return lfilter(b, a, data)

def normalize_16bits(data):
    return data.astype(np.float32) / 32768.0  # 16비트 정규화
