import matplotlib.pyplot as plt

from viterbi_decoding import *
from extract import ExtractPeaks

# load wave
file = 'signals/tags10_snr20_db.mat'
ep = ExtractPeaks(filename=file)

# extract edges from wave signal
rx_signal = ep.extract(thres=0.05, duration=2)
# ep.plotEdges(rx_signal, 400)

# MAIN FUNCTION
alpha = 144 * ep.sigma ** 4
res = viterbi(rx_signal, alpha)

# plot distribution density in the I-Q domain
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
LIM = 0.3
ep.plotAmp(axes=axes[0])
plotCDF(axes=axes[1], rx_signal=rx_signal, result_class=res)
fig.set_tight_layout('tight')
plt.show()
plt.close(fig)
