import random
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm


def viterbi(rx_signal: np.ndarray, alpha: float):
    # initialize
    peaks_num = np.shape(rx_signal)[0]
    path_to_keep = 5
    result_path = np.zeros((path_to_keep, peaks_num), dtype=int)  # 分类完成的数组
    prob_paths = np.zeros((path_to_keep,), dtype=np.float64)  # 当前路径的概率
    curr_path_num = 1  # number of current paths

    result_path[0, 0] = 1
    prob_paths[0] = 0

    for i in tqdm(range(1, peaks_num), desc="Classifying..."):  # loop of peaks
        tag_num = result_path.max()
        prob_matrix = np.zeros((curr_path_num, tag_num + 1), dtype=np.float64) - 1e7
        # conditional probability on (paths - class)
        for j in range(curr_path_num):  # loop of paths
            max_class = result_path[j].max()
            prob_matrix[j, 0: max_class + 1] = prob_paths[j] + F(rx_signal, result_path[j, :i], max_class, index=i,
                                                                 max_period=4, alpha=alpha)
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

    print("Classified tags number: %d" % result_path.max())
    return result_path[0, :]


def plotCDF(rx_signal: np.ndarray, result_class: np.ndarray, lim:int = 5):
    assert rx_signal.shape[0] == len(result_class)
    assert rx_signal.shape[1] == 3
    max_classes = int(result_class.max())
    colors = ['r', 'orange', 'yellow', 'green', 'blue', 'purple']
    colors.extend(
        ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(max_classes - 6)])
    for i in range(len(result_class)):
        plt.scatter(rx_signal[i, 1], rx_signal[i, 2], s=2, c=colors[result_class[i] - 1])
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.xlim((-lim, lim))
    plt.ylim((-lim, lim))
    plt.scatter([0], [0], marker='x', c='green')
    plt.show()
    plt.close()


if __name__ == "__main__":
    pass
