import os.path
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

# create configuration
save_path = 'tags5_noise1.npy'
config = {'time': 0.004, 'fs': 10000000, 'num': 6}
winlen = 4
noise = 0.3     # standard deviation of gaussian noise
edge_noise = 0.1

# simulation time (sec) && sampling rate (Hz) && number of tags
T, fs, num = config['time'], config['fs'], config['num']
t = np.linspace(0, T, int(T * fs))

# properties of square waveforms: frequency (Hz), amplitude, phase
freq = np.arange(num) * 15000 + 30000
amp = np.random.random(num) * 3 + 2 + 1j * (np.random.random(num) * 3 + 2)      # random amplitude
phase = np.random.random(num) * 2 * np.pi       # random phase
config['freq'] = freq       # add frequency to config
config['amplitude'] = amp       # add amplitude to config

# generate square waveforms
result = 0
for i in range(num):
    pwm = amp[i] * signal.square(2 * np.pi * freq[i] * t + phase[i])
    window = signal.windows.hamming(winlen) + np.random.normal(0, edge_noise, winlen)
    pwm = signal.convolve(pwm, window, mode='same')
    result += pwm

# add channel noise
result += np.random.normal(0, noise, len(t))
plt.plot(t[:200], result[:200], '.-')
plt.show()

# save wave to 'signals/'
filename = os.path.join('signals', save_path)
with open(filename, 'wb') as f:
    np.save(f, config, allow_pickle=True)
    np.save(f, result)
