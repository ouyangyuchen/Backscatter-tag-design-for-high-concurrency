from utils import *
from tqdm import tqdm


def viterbi(rx_signal: np.ndarray, alpha: float):
    # initialize
    peaks_num = np.shape(rx_signal)[0]
    path_to_keep = 8
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
            prob_matrix[j, 0: max_class + 1] = prob_paths[j] + F(rx_signal, result_path[j, :], max_class, index=i,
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

    return result_path[0, :]


def F(
        rx_signal: np.ndarray,
        original_path: np.ndarray,
        max_class: int,
        index: int,
        max_period: int,
        alpha: float,
):
    def distance(x1: np.ndarray, x2: np.ndarray):
        x2 = np.reshape(x2, (1, 2))
        d1 = np.sum((x1 - x2) ** 2, axis=1)
        d2 = np.sum((x1 + x2) ** 2, axis=1)
        d = np.minimum(d1, d2)
        return np.average(d)

    # find the indices of previous edges in the same class
    classes = find_pre_index(original_path, index, max_period)
    # get the average distance to the previous points
    distance_list = np.zeros((max_class,))
    for i in range(max_class):
        distance_list[i] = distance(rx_signal[classes[i]][:, 1:], rx_signal[index][1:])
    distance_list = np.log(distance_list)
    # new tag
    logM2 = np.sum(distance_list)
    # existing tags
    logM1 = -2 * distance_list + logM2 + np.log(alpha)
    # Add period information
    for i in range(max_class):
        if len(classes[i]) >= 2:
            p1 = rx_signal[index, 0] - rx_signal[classes[i][0], 0]
            p2 = rx_signal[classes[i][0], 0] - rx_signal[classes[i][1], 0]
            temp = max(p1 / p2, p2 / p1)
            if temp < 2.5:
                dp = abs(temp - np.around(temp))  # p1 / p2 is close to an integer?
                logM1[i] = logM1[i] + 1 * (1 - dp * 5)  # if dp < 0.x -> more likely, else less likely

    result_prob = np.zeros((max_class + 1,), dtype=np.float32)
    result_prob[:max_class] = logM1  # M1 - previous classes
    result_prob[max_class] = logM2  # M2 - new class
    return result_prob


if __name__ == "__main__":
    pass
