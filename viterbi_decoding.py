import numpy as np
from metric import F


def viterbi(rx_signal: np.ndarray):
    # initialize
    peaks_num = np.shape(rx_signal)[0]
    path_to_keep = 3
    result_path = np.zeros((path_to_keep, peaks_num), dtype=int)  # 分类完成的数组
    prob_paths = np.zeros((path_to_keep, 1), dtype=np.float32)  # 当前路径的概率
    curr_path_num = 1  # number of current paths

    result_path[0, 0] = 1
    prob_paths[0] = 1

    for i in range(1, peaks_num):  # loop of peaks
        peak = rx_signal[i]
        tag_num = result_path.max()
        prob_matrix = np.zeros(curr_path_num, tag_num + 1)
        for j in range(curr_path_num):  # loop of paths
            max_class = result_path[j].max()
            for k in range(1, max_class + 2):  # loop of class
                # suppose the current peak is classified to k
                possible_path = result_path[j, :]
                possible_path[i] = k
                prob_matrix[j, k] = prob_paths[j] * F(
                    rx_signal, possible_path, index=i, max_peiod=1
                )
        # find k max prob indices
        # indices = find_n_max(prob_matrix, path_to_keep)
        # update result_path and prob_paths based on indices


if __name__ == "__main__":
    peaks_num = 100
    rx_signal = np.zeros(shape=(peaks_num, 3), dtype=np.float32)
    rx_signal[:, 1:3] = np.random.random(size=(peaks_num, 2))
    viterbi(rx_signal)
