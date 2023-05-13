import matplotlib.pyplot as plt
import numpy as np

from utils import *
from scipy.io import loadmat


class ExtractPeaks:
    SHIFT = 2
    AWAY = 20

    def __init__(self, filename: str, start: int = 0, end: int = None) -> None:
        """
        Loading wave signal and other parameters

        :param filename: '.mat' file, must contains "wave", ("freq", "amp", "phases" are optional)
        :param n: number of points loaded from file, default is 0 (all)
        """
        file = loadmat(filename)
        self.wave = file["wave"].flatten()
        if end is not None:  # the proportion 'n' of wave
            self.wave = self.wave[start:end]

        temp = np.roll(self.wave, ExtractPeaks.SHIFT)
        self.impulses = (self.wave - temp)[ExtractPeaks.AWAY:]  # neglect the head elements

        # get variance of noise in real and imag parts
        thres = np.average(abs(self.impulses)) / 2
        self.filtered = self.impulses[abs(self.impulses) < thres]
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

    def extractRate(self, signal: np.ndarray, fs: float):
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
            return None
        if self.phases is None:
            self.phases = np.zeros((self.tags,))
        t_max = self.wave.size / fs
        lb = np.ceil(self.phases / np.pi)
        ub = np.floor((2 * np.pi * self.freq * t_max + self.phases) / np.pi)
        cnt = (ub - lb + 1).flatten()
        return cnt

    def plotAmp(self, axes: plt.Axes):
        if self.tags is None:
            return
        axes.scatter(np.real(self.amp), np.imag(self.amp), marker='o')
        axes.scatter(-np.real(self.amp), -np.imag(self.amp), marker='o')
        axes.scatter([0], [0], marker='x', c='green')
        axes.set_title("Amplitudes in I-Q domain")
        axes.set_xlabel("real")
        axes.set_ylabel("imag")
        axes.legend(['positive edges', 'negative edges'])

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
