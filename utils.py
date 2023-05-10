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
        if np.dot(x1[1:], x2[1:]) > 0:
            return pow(x1[1] - x2[1], 2) + pow(x1[2] - x2[2], 2) + 1e-5
        else:
            return pow(x1[1] + x2[1], 2) + pow(x1[2] + x2[2], 2) + 1e-5

    result_prob = np.zeros((max_class + 1,), dtype=np.float32)
    classes = np.zeros((max_class,), dtype=int) - 1
    # find the nearest edge for each class, start from the end
    cnt = 0
    for j in range(index - 1, -1, -1):
        tag = original_path[j]
        if classes[tag - 1] < 0:
            classes[tag - 1] = j
            cnt += 1
            if cnt >= max_class:
                break

    distance_list = np.zeros((max_class,))
    for i in range(max_class):
        distance_list[i] = distance(rx_signal[classes[i]], rx_signal[index])
    distance_list = np.log(distance_list)
    # new tag
    logM2 = np.sum(distance_list)
    # existing tags
    logM1 = -2 * distance_list + logM2 + np.log(alpha)
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


if __name__ == "__main__":
    pass
