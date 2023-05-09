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

    result_prob = np.zeros((max_class + 1, ), dtype=np.float32)
    classes = np.zeros((max_class,), dtype=int)

    for i, tag in enumerate(original_path):
        classes[tag - 1] = i

    distance_list = []
    for i in range(max_class):
        distance_list.append(distance(rx_signal[classes[i]], rx_signal[index]))
    distance_list_sum = sum(distance_list)
    # existing tags
    for tag_to_decide in range(max_class):
        result_prob[tag_to_decide] = distance_list_sum - distance_list[tag_to_decide] + \
                                     min(1 / distance_list[tag_to_decide], 1e5) * alpha
    # new tag
    result_prob[max_class] = distance_list_sum
    result_prob[:max_class + 1] += 1e-5 * np.max(result_prob)
    result_prob = result_prob / np.max(result_prob)
    return result_prob


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


if __name__ == "__main__":
    pass
