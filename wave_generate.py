import math
import os.path
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import scipy

# create configuration
config = {'save_path': 'tags5_noise1.npy',
          'time': 0.002, 'fs': 1000000, 'num': 5}

# simulation time (sec) && sampling rate (Hz) && number of tags
T, fs, num = config['time'], config['fs'], config['num']
t = np.linspace(0, T, int(T * fs))

# properties of square waveforms: frequency (Hz), amplitude, phase
freq = np.arange(num) * 200 + 500
amp = np.random.random(num) * 3 + 2 + 1j * (np.random.random(num) * 3 + 2)
phase = np.random.random(num) * 2 * np.pi
config['freq'] = freq
config['amplitude'] = amp

# generate square waveforms
result = 0
for i in range(num):
    pwm = amp[i] * signal.square(2 * np.pi * freq[i] * t + phase[i])
    window = signal.windows.hamming(math.floor(3))
    pwm = signal.convolve(pwm, window, mode='same')
    result += pwm

# plot result in time and frequency domain
fig, axes = plt.subplots(2, 2, sharex='col')
axes[0, 0].plot(abs(scipy.fft.fft(result)))
axes[0, 0].set_title('No gaussian noise')

axes[0, 1].plot(t, abs(result))

result += np.random.normal(0, 0.5, len(t))
axes[1, 0].plot(abs(scipy.fft.fft(result)))
axes[1, 0].set_title('With gaussian noise')

axes[1, 1].plot(t, abs(result))

plt.show()

# save wave to 'signals/'
filename = os.path.join('signals', config['save_path'])
with open(filename, 'wb') as f:
    np.save(f, config, allow_pickle=True)
    np.save(f, result)
