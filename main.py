from viterbi_decoding import *
from extract import ExtractPeaks

file = 'signals/tags4_snr40_db.mat'
ep = ExtractPeaks(filename=file)
# extract edges from wave signal
rx_signal = ep.extract(thres=5.0, duration=2)

# determine alpha, (0 < alpha < 1/2)
alpha = 2000

# MAIN FUNCTION
res = viterbi(rx_signal, alpha)

# plot distribution density in the I-Q domain
plotCDF(rx_signal, res)
