import random
import matplotlib.pyplot as plt
from utils import *
from extract import ExtractPeaks
from tqdm import tqdm


def viterbi(rx_signal: np.ndarray, alpha: float):
    # initialize
    peaks_num = np.shape(rx_signal)[0]
    path_to_keep = 20
    result_path = np.zeros((path_to_keep, peaks_num), dtype=int)  # 分类完成的数组
    prob_paths = np.zeros((path_to_keep,), dtype=np.float32)  # 当前路径的概率
    curr_path_num = 1  # number of current paths

    result_path[0, 0] = 1
    prob_paths[0] = 1

    for i in tqdm(range(1, peaks_num), desc="Classifying..."):  # loop of peaks
        tag_num = result_path.max()
        prob_matrix = np.zeros((curr_path_num, tag_num + 1), dtype=np.float32)
        for j in range(curr_path_num):  # loop of paths
            max_class = result_path[j].max()

            original_path = np.array(result_path[j, :])
            prob_matrix[j, 0: max_class + 1] = prob_paths[j] * F(rx_signal, original_path[0: i], max_class, index=i,
                                                                 max_period=1, alpha=alpha)

        # find k max prob indices
        indices = find_n_max(prob_matrix, path_to_keep)
        curr_path_num = len(indices)
        # update result_path and prob_paths based on indices
        temp_res = np.zeros((curr_path_num, i + 1), dtype=int)
        for j in range(curr_path_num):
            temp_res[j, :i] = result_path[indices[j][0], :i]
            temp_res[j, i] = indices[j][1] + 1
            prob_paths[j] = prob_matrix[indices[j]]
        result_path[:curr_path_num, :i + 1] = temp_res
        normalize(prob_paths)

    print("Classified tags number: %d" % result_path.max())
    return result_path[0, :]


def plotCDF(rx_signal: np.ndarray, result_class: np.ndarray):
    assert rx_signal.shape[0] == len(result_class)
    assert rx_signal.shape[1] == 3
    max_classes = int(result_class.max())
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(max_classes)]
    for i in range(len(result_class)):
        plt.scatter(rx_signal[i, 1], rx_signal[i, 2], s=4, c=colors[result_class[i] - 1])
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.show()
    plt.close()


if __name__ == "__main__":
    ep = ExtractPeaks(filename='signals/tags20_snr40_db.mat')

    rx_signal = ep.extract()

    # determine alpha, according to the credibility
    alpha = 10000 * ep.sigma ** 2
    print(alpha)
    res = viterbi(rx_signal, alpha)
    plotCDF(rx_signal, res)
