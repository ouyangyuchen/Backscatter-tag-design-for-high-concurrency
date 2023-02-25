import os.path
import numpy as np
import scipy
from scipy import signal
from matplotlib import pyplot as plt


def loadWave(filename: str, lowpass: bool = False, window_size: int = 3):
    with open(filename, 'rb') as f:
        config = np.load(f, allow_pickle=True)
        result = np.load(f)
    if lowpass:
        hamming = signal.windows.hamming(window_size)
        result = signal.convolve(result, hamming, mode='same')
    return config, result


if __name__ == '__main__':
    wavefile = 'tags5_noise1.npy'
    _, waveform = loadWave(os.path.join('signals', wavefile), lowpass=False)
    fig, axes = plt.subplots(2, 1, sharey='col')
    # axes[0, 0].plot(waveform)
    axes[0].plot(waveform - np.roll(waveform, 3))
    _, waveform = loadWave(os.path.join('signals', wavefile), window_size=5, lowpass=True)
    # axes[1, 0].plot(waveform)
    axes[1].plot(waveform - np.roll(waveform, 3))
    plt.show()
