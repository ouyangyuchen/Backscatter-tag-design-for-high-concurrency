import numpy as np
from matplotlib import pyplot as plt
import random


def find_n_max(prob_matrix: np.ndarray, path_to_keep: int):
    prob_temp = np.array(prob_matrix)
    result_index = []
    for _ in range(path_to_keep):
        m_index = np.argmax(prob_temp, axis=None)
        m_position = np.unravel_index(m_index, np.shape(prob_temp))
        m = prob_temp[m_position]
        if m < -1e6:
            break
        else:
            result_index.append(m_position)
            prob_temp[m_position] = -1e7

    return result_index


def find_pre_index(arr: np.ndarray, index: int, max_period: int):
    max_class = arr.max()
    res = [[] for _ in range(max_class)]
    cnt = np.zeros((max_class,), dtype=int)
    full = np.zeros((max_class,)).astype(bool)

    for i in range(index - 1, -1, -1):
        tag = arr[i] - 1
        if not full[tag]:
            res[tag].append(i)
            cnt[tag] += 1
            full[tag] = cnt[tag] == max_period
        if full.sum() == max_class:
            break
    return res


def plotCDF(axes: plt.Axes, rx_signal: np.ndarray, result_class: np.ndarray, snr: int = None):
    assert rx_signal.shape[0] == len(result_class)
    assert rx_signal.shape[1] == 3
    max_classes = int(result_class.max())
    colors = ['#f7f7f7', 'r', 'orange', 'yellow', 'green', 'blue', 'purple']
    colors.extend(
        ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(max_classes - 6)])
    axes.scatter([0], [0], marker='x', c='black')
    for i in range(len(result_class)):
        plt.scatter(rx_signal[i, 1], rx_signal[i, 2], s=2, c=colors[result_class[i]])
    axes.set_xlabel("Real")
    axes.set_ylabel("Imag")
    if snr is not None:
        axes.set_title("Classified results  SNR=%d(dB)" % snr)
    else:
        axes.set_title("Classified Results in I-Q domain")


def filtering(rx_signal: np.ndarray, result_class: np.ndarray):
    assert rx_signal.shape[0] == len(result_class)
    assert rx_signal.shape[1] == 3
    max_classes = result_class.max()
    cnt = np.zeros((max_classes,))
    for i in range(len(result_class)):
        cnt[result_class[i] - 1] += 1
    valid_classes = np.argwhere(cnt > 0.01 * len(result_class)) + 1
    valid_classes = set(valid_classes.flatten().tolist())
    # result_class[invalid_edge] = 0
    for i in range(len(result_class)):
        if result_class[i] not in valid_classes:
            result_class[i] = 0
    return valid_classes


def get_freq(rx_signal: np.ndarray, result_path: np.ndarray):
    indices = dict()
    for i, tag in enumerate(result_path):
        if tag != 0:
            indices.setdefault(tag, []).append(int(rx_signal[i, 0]))
    # 1st order diff
    dn = dict()
    for tag in indices.keys():
        temp = np.array(indices[tag], dtype=int)
        dn[tag] = np.median(np.diff(temp))
    # from dn to freq
    freq = dict()
    fs = 1e+7
    for tag in dn.keys():
        freq[tag] = fs / dn[tag] / 2
    return freq, indices


def freq_match(freq: dict, gt_freq: np.ndarray, df: float):
    """Find the closest freq in ground truth for each tag in freq."""
    res = [[] for _ in range(len(gt_freq))]
    for tag in freq.keys():
        i = np.argmin(np.abs(gt_freq - freq[tag]))
        if abs(freq[tag] - gt_freq[i]) < df:
            res[i].append(tag)
    return res


def count_acc(match_list: list[list], gt_num: np.ndarray, indices: dict):
    assert len(match_list) == len(gt_num)
    detected_tags = 0
    edges_ratio = np.zeros((len(match_list),), dtype=float)
    for i in range(len(match_list)):  # tag index in gt
        if len(match_list[i]) > 0:
            detected_tags += 1
            for j in match_list[i]:
                edges_ratio[i] = edges_ratio[i] + len(indices[j])
    edges_ratio = edges_ratio / gt_num
    return detected_tags, edges_ratio


if __name__ == "__main__":
    pass
