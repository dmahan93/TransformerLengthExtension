from transformers import Trainer, GPTNeoXModel
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch


def causal_masking_with_eot(data: torch.tensor, eot_token: int):
    """
    This function takes in a numpy array of shape (batch_size, sequence_length) and creates a causal mask for each
    sequence in the batch. The mask is created by finding the index of the eot_token and setting all values in the
    mask after that index to 0. The mask is then applied to the data by multiplying the mask by the data.
    This ends up with independent triangle masks for each sequence.
    :param data: Tokenized data of shape (batch_size, sequence_length)
    :param eot_token: Token that indicates the end of a sequence
    :return: Attention mask of shape (batch_size, sequence_length, sequence_length)
    """
    mask = torch.tril(torch.ones((1, data.shape[0], data.shape[0]))).numpy()
    eots = np.argwhere(data == eot_token)
    for item in eots:
        mask[0][item[0]:, :item[0]] = 0
    return torch.from_numpy(mask).bool()


class XlDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        j = index // len(self.data[0])
        i = index % len(self.data[0])
        if j == 0:
            return [
                torch.tensor(self.data[i, j*self.seq_len:(j+1)*self.seq_len]),
                causal_masking_with_eot(torch.tensor(self.data[i, j*self.seq_len:(j+1)*self.seq_len]), 0)
                ]
        else:
            return [
                torch.tensor(self.data[i, j*self.seq_len:(j+1)*self.seq_len]),
                causal_masking_with_eot(torch.tensor(self.data[i, (j-1)*self.seq_len:(j+1)*self.seq_len]), 0)[:, self.seq_len:, :]
                ]

    def __len__(self):
        return len(self.data) * len(self.data[0])//self.seq_len


def XlDatasetCollateFn(data, eot_token=0):
    return {
        "input_ids": torch.stack([f[0] for f in data]),
        "attention_mask": torch.stack([f[1] for f in data])
    }


class XlTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)