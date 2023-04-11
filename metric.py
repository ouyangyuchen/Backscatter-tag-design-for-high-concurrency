import numpy as np


def F(
    rx_signal: np.ndarray,
    possible_path: np.ndarray,
    index: int,
    max_peiod: int,
    alpha: float,
):
    def distance(x1: np.ndarray, x2: np.ndarray):
        if np.dot(x1[1:], x2[1:]) > 0:
            return pow(x1[1] - x2[1], 2) + pow(x1[2] - x2[2], 2)
        else:
            return pow(x1[1] + x2[1], 2) + pow(x1[2] + x2[2], 2)

    tag_to_decide = possible_path[index]
    original_path = possible_path[0:index]
    original_tag_num = original_path.max()

    # find the last indices of each class in the orignal path
    classes = np.zeros((original_tag_num,), dtype=int)
    for i, tag in enumerate(original_path):
        classes[tag - 1] = i

    p = 0.0
    if tag_to_decide <= original_tag_num:
        # classified to old tags
        for i in range(original_tag_num):
            if (i + 1) == tag_to_decide:
                p += alpha / distance(rx_signal[classes[i]], rx_signal[index])
            else:
                p += distance(rx_signal[classes[i]], rx_signal[index])
    else:
        # classified to the new tag
        for i in range(original_tag_num):
            p += distance(rx_signal[classes[i]], rx_signal[index])
    return p


if __name__ == "__main__":
    peaks_num = 100
    rx_signal = np.zeros(shape=(peaks_num, 3), dtype=np.float32)
    rx_signal[:, 1:3] = np.random.random(size=(peaks_num, 2))