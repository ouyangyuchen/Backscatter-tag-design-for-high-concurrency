import numpy as np
import scipy.io


def loadWave(filename: str, is_matlab=True) -> np.ndarray:
    """
    Load signal from file and extract the raw edges.
    """
    # load wave signal into python
    if is_matlab:
        wave = scipy.io.loadmat(filename)["wave"][0, :]
    else:
        wave = np.load(filename)
    # substract with its shift signal
    temp = np.roll(wave, 1)
    temp[:1] = complex(0.0, 0.0)
    impulses = (wave - temp)[20:-20]
    return impulses


def extractPeaks(signal: np.ndarray, thres: float):
    """Return nx3 matrix from the peaks of raw signal."""
    indices = np.argwhere(np.abs(signal) >= thres)

    edges = []
    for i in indices:
        if len(edges) == 0 or i - edges[-1] > 2:
            edges.append(i)
        elif abs(signal[i]) > abs(signal[edges[-1]]):
            edges[-1] = i
    edges = np.array(edges)
    real_part = np.real(signal[edges])
    imag_part = np.imag(signal[edges])
    return np.concatenate((edges, real_part, imag_part), axis=1)


def find_n_max(prob_matrix: np.ndarray, path_to_keep: int):
    prob_temp = np.array(prob_matrix)
    result_index = []
    for _ in range(path_to_keep):
        m_index = np.argmax(prob_temp, axis=None)
        m_position = np.unravel_index(m_index, np.shape(prob_temp))
        m = prob_temp[m_position]
        if m == 0.0:
            break
        else:
            result_index.append(m_position)
            prob_temp[m_position] = 0.0

    return result_index


def normalize(arr: np.ndarray):
    arr /= arr.sum()


if __name__ == "__main__":
    signal = loadWave("./signals/tags20_noise_0.50_2000.mat")
    extractPeaks(signal, thres=5)
