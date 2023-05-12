from viterbi_decoding import *
from extract import ExtractPeaks

# load wave
snr = 25
tags = 15
file = 'signals/tags%d_snr%d_db.mat' % (tags, snr)
ep = ExtractPeaks(filename=file, start=0, end=None)
print("sigma \t %.4f" % ep.sigma)

# extract edges from wave signal
rx_signal = ep.extract(thres=0.15, duration=3)
# ep.plotEdges(rx_signal, 400)

# MAIN FUNCTION
alpha = 338.56 * ep.sigma ** 4
res = viterbi(rx_signal, alpha)
valid_class = filtering(rx_signal, res)
print("tags: %d" % len(valid_class))

# test performance
freq = get_freq(rx_signal, res)
tags_match = freq_match(freq, ep.freq, df=100)
print(tags_match)

# plot distribution density in the I-Q domain
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
ep.plotAmp(axes=axes[0])
plotCDF(axes=axes[1], rx_signal=rx_signal, result_class=res, snr=snr)
fig.set_tight_layout('tight')
plt.show()
plt.close(fig)
