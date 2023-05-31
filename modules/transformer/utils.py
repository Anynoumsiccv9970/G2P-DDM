
import torch
import torch.nn as nn
from torch import Tensor
import math


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


if __name__ == "__main__":
    def window_subsequent_mask(size, window_size):
        pos = torch.arange(0, size).unsqueeze(0).repeat(size, 1)
        right = torch.arange(window_size-1, size, window_size).unsqueeze(1).repeat(1, window_size).view(size, 1)
        mask = (pos <= right)
        return mask.unsqueeze(0)

    def delay_subsequent_mask(size, teacher_len):
        pos = torch.arange(0, size).unsqueeze(0).repeat(size, 1)
        left = (torch.arange(0, size) - teacher_len).unsqueeze(1)
        mask = (pos <= left).long() + torch.eye(size).long()
        return mask.unsqueeze(0)
    
    mask = delay_subsequent_mask(15, 4)
    print(mask)

