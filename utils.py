import numpy as np


def find_n_max(prob_matrix: np.ndarray, path_to_keep: int):
    prob_temp = np.array(prob_matrix)
    result_index = []
    for i in range(path_to_keep):
        m_index = np.argmax(prob_temp, axis=None)
        m_position = np.unravel_index(m_index, np.shape(prob_temp))
        m = prob_temp[m_position]
        if m == 0.0:
            break
        else:
            result_index.append(m_position)
            prob_temp[m_position] = 0.0

    return result_index

def normalize(arr:np.ndarray):
    arr /= arr.sum()


if __name__ == '__main__':
    path_to_keep = 3
    prob_matrix = np.zeros(shape=(2, 5), dtype=np.float32)
    prob_matrix[:, :] = np.random.random(size=(2, 5))
    print(prob_matrix)
    print(find_n_max(prob_matrix, path_to_keep))
