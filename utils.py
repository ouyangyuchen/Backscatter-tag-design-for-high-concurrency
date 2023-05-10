import numpy as np


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

    classes = find_pre_index(original_path, index, max_period)

    distance_list = np.zeros((max_class,))
    for i in range(max_class):
        distance_list[i] = distance(rx_signal[classes[i]][:, 1:], rx_signal[index][1:])
    distance_list = np.log(distance_list)
    # new tag
    logM2 = np.sum(distance_list)
    # existing tags
    logM1 = -2 * distance_list + logM2 + np.log(alpha)
    result_prob = np.zeros((max_class + 1,), dtype=np.float32)
    result_prob[:max_class] = logM1  # M1 - previous classes
    result_prob[max_class] = logM2  # M2 - new class
    return result_prob


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


if __name__ == "__main__":
    pass
