import numpy
import torch
import numpy as np
import matplotlib.pyplot as plt


def mask_examine():
    mask = torch.tril(torch.ones((2048, 2048))).numpy()
    plt.imshow(mask)
    plt.show()


def causal_masking_with_eot(data: numpy.ndarray, eot_token: int):
    """
    This function takes in a numpy array of shape (batch_size, sequence_length) and creates a causal mask for each
    sequence in the batch. The mask is created by finding the index of the eot_token and setting all values in the
    mask after that index to 0. The mask is then applied to the data by multiplying the mask by the data.
    This ends up with independent triangle masks for each sequence.
    :param data: Tokenized data of shape (batch_size, sequence_length)
    :param eot_token: Token that indicates the end of a sequence
    :return: Attention mask of shape (batch_size, sequence_length, sequence_length)
    """
    mask = torch.tril(torch.ones((data.shape[0], data.shape[1], data.shape[1]))).numpy()
    eots = np.argwhere(data == eot_token)
    for item in eots:
        mask[item[0]][item[1]:, :item[1]:] = 0
    return mask


if __name__ == "__main__":
    # mask_examine()
    data = np.ones((2, 2048))
    data[0][22] = 0
    data[0][444] = 0
    data[0][1023] = 0
    data[1][123] = 0
    data[1][201] = 0
    data[1][1965] = 0
    causal_masking_with_eot(
        data,
        0
    )