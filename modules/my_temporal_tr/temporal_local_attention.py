import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import math
import torch
import torch.nn as nn
from torch import Tensor
from .relative_deberta import DisentangledSelfAttention
from .relative_local_deberta import DisentangledLocalSelfAttention
from .position_encoding import PositionalEncoding
import random

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(self, opts, freeze: bool = False, **kwargs):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(TransformerEncoder, self).__init__()

        # build all (num_layers) layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    opts=opts,
                    size=opts.hidden_size,
                    ff_size=opts.ff_size,
                    num_heads=opts.num_heads,
                    dropout=opts.dropout,
                    local_layer=num < opts.local_num_layers,
                    window_size=(num + 1) * opts.window_size
                )
                for num in range(opts.num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(opts.hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(opts.hidden_size)
        self.emb_dropout = nn.Dropout(p=opts.emb_dropout)
        self._output_size = opts.hidden_size


    # pylint: disable=arguments-differ
    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        x = self.pe(embed_src)  # add position encoding to word embeddings
        x = self.emb_dropout(x)  # [bs, length, embed_size]

        for layer in self.layers:
            x = layer(x, mask, src_length)
        return self.layer_norm(x)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
        self, opts, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1,
        local_layer: bool = False, window_size: int = 0, spatial_attn: bool = False,
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()
        self.window_size = window_size

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.rel_embeddings = nn.Embedding(opts.max_relative_positions * 2, size)
        
        if local_layer:
            # print("window_size: ", self.window_size)
            self.src_src_att = DisentangledLocalSelfAttention(opts, size, num_heads, 
                                                              window_size=self.window_size, 
                                                              spatial_attn=spatial_attn)
        else:
            self.src_src_att = DisentangledSelfAttention(opts)

        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor, src_length: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        x_norm = self.layer_norm(x)
        # print("x_norm: ", x.shape)

        if mask is not None:
            mask = self.get_attention_mask(mask)
        # print("mask: ", mask.shape)

        h = self.src_src_att(
            hidden_states=x_norm,
            attention_mask=mask,
            return_att=False,
            query_states=None,
            relative_pos=None,
            rel_embeddings=self.rel_embeddings.weight)

        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # print("extended_attention_mask: ", extended_attention_mask.shape)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.byte()
            # print("attention_mask: ", attention_mask.shape, attention_mask)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask


if __name__ == "__main__":
    class Config():
        hidden_size = 512
        ff_size = 2048
        num_heads = 8
        dropout = 0.1
        emb_dropout = 0.1
        num_layers = 6
        local_num_layers = 3
        use_relative = True
        max_relative_positions = 32
        window_size = 16

    opts = Config()
    m = TransformerEncoder(opts)
    x = torch.randn(5, 100, 512)
    mask = torch.randint(0, 100, (5, 100)).ne(0)
    x_len = mask.sum(-1)
    out = m(x, x_len, mask)
    print("out: ", out.shape)