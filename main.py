from viterbi_decoding import *
from extract import ExtractPeaks

file = 'signals/tags10_snr40_db.mat'
ep = ExtractPeaks(filename=file)

LIM = 20
ep.plotAmp(lim=LIM)

# extract edges from wave signal
rx_signal = ep.extract(thres=5.0, duration=2)
ep.plotEdges(rx_signal, 400)

alpha = 1e+4

# MAIN FUNCTION
res = viterbi(rx_signal, alpha)

# plot distribution density in the I-Q domain
plotCDF(rx_signal, res, lim=2 * LIM)
