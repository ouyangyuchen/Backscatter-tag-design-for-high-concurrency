import os.path
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt


def loadWave(filename: str, lowpass: bool = False, window_size: int = 3):
    with open(filename, 'rb') as f:
        config = np.load(f, allow_pickle=True)
        result = np.load(f)
    return config, result


def deNoise(wave_arr: np.ndarray, shift: int = 1):
    # cfreq = np.array([100000]) * 2 * np.pi
    # sos = signal.butter(N=1, Wn=cfreq, btype='lp', fs=10000000, output='sos')
    # wave_arr = signal.sosfilt(sos, wave_arr)
    wave_shift = np.roll(wave_arr, shift)
    wave_shift[:shift] = 0.0
    wave_arr -= wave_shift
    return wave_arr


if __name__ == '__main__':
    fig, axes = plt.subplots(2, 1)
    wavefile = 'tags5_noise1.npy'
    config, wavearray = loadWave(os.path.join('signals', wavefile))
    axes[0].plot(wavearray[:500])
    axes[0].set_title('square waveform')

    wavearray = deNoise(wavearray, shift=2)
    axes[1].plot(wavearray[:500])
    axes[1].set_title('extract edges')
    plt.tight_layout()
    plt.show()
