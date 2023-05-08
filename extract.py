import matplotlib.pyplot as plt
import numpy as np

from utils import *
from scipy.io import loadmat


class ExtractPeaks:
    SHIFT = 1
    AWAY = 20
    MIN_SNR = 10  # the lower bound of SNR (unit: gain, not dB)

    def __init__(self, filename: str) -> None:
        """
        Loading wave signal and other parameters

        Parameters
        ---
        filename: '.mat' file, must contains "wave", ("freq", "amp", "phases" are optional)
        """
        file = loadmat(filename)
        self.wave = file["wave"].flatten()

        temp = np.roll(self.wave, ExtractPeaks.SHIFT)
        temp[:ExtractPeaks.SHIFT] = complex(0.0, 0.0)
        self.impulses = (self.wave - temp)[ExtractPeaks.AWAY:]  # neglect the head elements

        # get variance of noise in real and imag parts
        self.filtered = self.impulses[abs(self.impulses) < np.average(abs(self.impulses)) / 2]
        noises = np.concatenate([np.real(self.filtered), np.imag(self.filtered)])
        self.sigma = np.sqrt(np.average(noises ** 2) / 2)

        self.freq, self.amp, self.phases, self.tags = None, None, None, None
        if file.__contains__("freq"):
            self.freq = file["freq"][0, :]
            self.tags = self.freq.size
        if file.__contains__("amp"):
            self.amp = file["amp"][0, :]
            assert self.tags == self.amp.size
        if file.__contains__("phases"):
            self.phases = file["phases"][0, :]
            assert self.tags == self.phases.size

    def extract(self, thres: float = 5.0, duration: int = 2):
        """Extract the peaks in the signal.

        Parameters
        ---
        thres: float
            the amplitude will be taken into account only if amp[i] > thres
        duration: int
            the amplitude will be seemed as a peak only if it is bigger than
            any in [i-duration, i+duration]

        Return
        ---
        np.ndarray[n][3]
            indices | real amplitudes | imag amplitudes
        """
        # extract the transition edges in the signal
        indices = np.argwhere(np.abs(self.impulses) >= thres)
        edges = []
        for i in indices:
            if len(edges) == 0 or i - edges[-1] > duration:
                edges.append(i)
            elif np.abs(np.angle(self.impulses[i]) - np.angle(self.impulses[edges[-1]])) > 0.15:
                edges.append(i)
            elif abs(self.impulses[i]) > abs(self.impulses[edges[-1]]):
                edges[-1] = i
        edges = np.array(edges)
        real_part = np.real(self.impulses[edges])
        imag_part = np.imag(self.impulses[edges])
        res = np.concatenate((edges, real_part, imag_part), axis=1)
        return res

    def extractRate(self, signal: np.ndarray, fs: float) -> float:
        """number of edges extracted / number of total edges.

        Available only when the input file has "freq", "phases".

        Parameters
        ---
        signal: np.ndarray[n][3]
            the result matrix returned from extract().
        fs: float
            sampling frequency of the wave signal.
        """
        assert signal.shape[1] == 3
        if self.tags is None:
            raise ValueError(
                "The input matlab data must contain phases and frequencies."
            )
        t_max = self.wave.size / fs
        lb = np.ceil(self.phases / np.pi)
        ub = np.floor((2 * np.pi * self.freq * t_max + self.phases) / np.pi)
        cnt = np.sum(ub - lb + 1)
        return signal.shape[0] / cnt

    # def extractAcc(self, signal: np.ndarray, fs: float) -> float:
    #     """The proportion of valid edges in the extracted array. Valid
    #     edges correspond to exactly one transition edge among all tags.
    #
    #     Available only when the input file has "freq", "phases".
    #
    #     Parameters
    #     ---
    #     signal: np.ndarray[n][3]
    #         the result matrix returned from extract().
    #     fs: float
    #         sampling frequency of the wave signal.
    #     """
    #     assert signal.shape[1] == 3
    #     if self.tags is None:
    #         raise ValueError(
    #             "The input matlab data must contain phases and frequencies."
    #         )
    #     time = (signal[:, 0] + ExtractPeaks.away) / fs
    #     tolerant = 2 * self.freq * (ExtractPeaks.shift + 1) / fs
    #     cnt = 0
    #     for t in time:
    #         phase = 2 * self.freq * t + self.phases / np.pi
    #         std = np.abs(phase - np.around(phase))
    #         std = std <= tolerant
    #         if std.sum() == 1:
    #             cnt += 1
    #     return cnt / time.size

    def plotEdges(self, rx_signal: np.ndarray, n: int):
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(self.wave[0:n])
        axes[1].plot(self.impulses[0:n])
        edges = rx_signal[rx_signal[:, 0] < n, :]
        axes[1].scatter(edges[:, 0], edges[:, 1], c='r')
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    filename = "signals/tags20_snr30_db.mat"
    ep = ExtractPeaks(filename)
    rx_signal = ep.extract(thres=5)
    ep.plotEdges(rx_signal, n=200)
