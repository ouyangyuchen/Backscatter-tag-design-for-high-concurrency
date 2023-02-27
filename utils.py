import os.path
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def loadWave(filename: str, lowpass: bool = False, window_size: int = 3):
    with open(filename, 'rb') as f:
        config = np.load(f, allow_pickle=True)
        result = np.load(f)
    return config, result


def deNoise(wave_arr: np.ndarray, window_size: int = 3):
    wave_shift = np.roll(wave_arr, window_size)
    wave_shift[:window_size] = 0.0
    wave_arr -= wave_shift
    hamming = signal.windows.hamming(window_size)
    wave_arr = signal.convolve(wave_arr, hamming, mode='same')
    return wave_arr[window_size:]


if __name__ == '__main__':
    fig, axes = plt.subplots(2, 1)
    wavefile = 'tags5_noise1.npy'
    _, wavearray = loadWave(os.path.join('signals', wavefile))
    axes[0].plot(wavearray[100: 500])

    wavearray = deNoise(wavearray, window_size=3)
    axes[1].plot(wavearray[100: 500])

    plt.show()
