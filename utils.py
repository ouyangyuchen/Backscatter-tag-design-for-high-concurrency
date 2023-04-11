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
    n = signal.shape[0]

    edges = np.zeros((n, 3), dtype=np.float64)
    cnt = 0
    for i in range(n):
        if np.abs(signal[i]) > thres:
            edges[cnt, 0] = i
            edges[cnt, 1] = np.real(signal[i])
            edges[cnt, 2] = np.imag(signal[i])
            cnt += 1
    return edges[:cnt, :]


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
    path_to_keep = 3
    prob_matrix = np.zeros(shape=(2, 5), dtype=np.float32)
    prob_matrix[:, :] = np.random.random(size=(2, 5))
    print(prob_matrix)
    print(find_n_max(prob_matrix, path_to_keep))
