import math
from tkinter import N
import torch
import torch.nn as nn
from torch import Tensor



# pylint: disable=arguments-differ
class PositionalEncoding(nn.Module):

    def __init__(self, size: int = 0, max_len: int = 5000):
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, x, past_length=None):
        # Add position encodings
        if past_length is not None:
            return self.pe[:, :x.size(1)]
        else:
            return self.pe[:, past_length:x.size(1)]


class LearnedPositionalEmbedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=None)

    def forward(self, input, is_points=False, token_num=None):
        if not is_points:
            positions = make_positions(
                input.data, self.padding_idx, is_points
            )
            # print("positions: ", is_points, positions)
            return self.embedding(positions)
        else:
            assert token_num is not None
            positions = make_positions(
                input.data, self.padding_idx, is_points
            )
            # print("positions: ", is_points, positions)
            positions = (positions - 1) // token_num
            # print("positions: ", is_points, positions[:, 75: 85], positions[:, 15:21])
            return self.embedding(positions)



def make_positions(tensor, padding_idx, is_points=True):
    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long()


if __name__ == "__main__":
    x = torch.LongTensor([[12,13,14,15,19,18,17,20, 20], [12,13,14,15,19,18,17,20, 20]])
    padding_idx = 20

    learn_embed = LearnedPositionalEmbedding(21, 64, padding_idx)
    out1 = learn_embed(x)
    print(out1.shape)
    out2 = learn_embed(x, True, 3)
    print(out2.shape)
