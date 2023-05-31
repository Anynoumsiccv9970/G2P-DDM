import math
import torch
import torch.nn as nn
from torch import Tensor



class MultiHeadedAttention3D(nn.Module):
    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        super(MultiHeadedAttention3D, self).__init__()

        assert size % num_heads == 0

        self.size = size
        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):


        bs, h, q_len, dim = q.size()
        if k.ndim != q.ndim:
            assert k.ndim == 3, q.ndim == 4
            k = v = k.unsqueeze(1)

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        # k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        # v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        # q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2) # [bs, head, length, hid_size]

        # compute scores
        q = q / math.sqrt(self.size)

        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            mask = self.get_attention_mask(mask)
            scores = scores.masked_fill(~mask, float("-inf"))
        
        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)  # [bs, head, length, length]

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        
        context = torch.matmul(attention, v)

        output = self.output_layer(context)

        return output

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # print("extended_attention_mask: ", extended_attention_mask.shape)
            # attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            # attention_mask = attention_mask.byte()
            # print("attention_mask: ", attention_mask.shape, attention_mask)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask



# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2) # [bs, head, length, hid_size]

        # compute scores
        q = q / math.sqrt(self.head_size)


        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            mask = self.get_attention_mask(mask)
            scores = scores.masked_fill(~mask, float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)  # [bs, head, length, length]

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )
        output = self.output_layer(context)
        return output

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # print("extended_attention_mask: ", extended_attention_mask.shape)
            # attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            # attention_mask = attention_mask.byte()
            # print("attention_mask: ", attention_mask.shape, attention_mask)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask


    def forward_fast(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None, layer_past=None):
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2) # [bs, head, length, hid_size]

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # compute scores
        q = q / math.sqrt(self.head_size)


        scores = torch.matmul(q, k.transpose(2, 3)) # [bs, head, q_len, kv_len]

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            mask = self.get_attention_mask(mask)
            scores = scores.masked_fill(~mask, float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)  # [bs, head, length, length]

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )
        output = self.output_layer(context)

        return output, present



if __name__ == "__main__":
    q = k = v = torch.randn(5, 10, 512)
    mask = None
    rl_pe = torch.randn(10, 10, 64)
    m = MultiHeadedAttention(num_heads=8, size=512, dropout=0.0)
    out = m(q,k,v, mask, rl_pe)
    print(out.shape)