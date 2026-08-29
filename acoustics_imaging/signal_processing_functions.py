import numpy as np
from scipy.fft import fft, ifft, fftfreq


def blackman_window(signal, center_arr, wavelength):
    window = np.zeros_like(signal)
    for idx, center in enumerate(center_arr):
        window[idx, center - wavelength:center + wavelength] = np.blackman(2 * wavelength)
    windowed_signal = window * signal
    return fft(windowed_signal)


def synthesize_signal(signal, peaks_arr, abs_max, mic_index):
    delay = np.int32(np.array(peaks_arr) - abs_max)
    signal[:] = signal[mic_index]
    signal = np.array([np.roll(signal[i], delay[i]) for i in range(len(delay))])
    return signal


def low_pass_filter(signal, sampling_period, fc):
    fs = 1 / sampling_period
    f = fftfreq(len(signal[0])) * fs
    signal[:, np.abs(f) > fc] = 0
    return np.real(ifft(signal))
