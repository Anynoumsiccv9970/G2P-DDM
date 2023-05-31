import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from modules.transformer.multihead_attention import MultiHeadedAttention
from modules.transformer.position_encoding import PositionalEncoding
from modules.transformer.encoder import PositionwiseFeedForward
import numpy as np
from modules.transformer.word_embedding import WordEmbeddings
from .utils import BertLayerNorm



def window_subsequent_mask(size, window_size):
    pos = torch.arange(0, size).unsqueeze(0).repeat(size, 1)
    right = torch.arange(window_size-1, size, window_size).unsqueeze(1).repeat(1, window_size).view(size, 1)
    mask = (pos <= right)
    return mask.unsqueeze(0)

def subsequent_mask(size: int) -> Tensor:
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.size = size
        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )

        self.x_layer_norm = BertLayerNorm(size, eps=1e-6)
        self.dec_layer_norm = BertLayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor = None, memory: Tensor = None, src_mask: Tensor = None, trg_mask: Tensor = None):        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)
        o = self.feed_forward(self.dropout(h2) + h1)

        return o

    def forward_fast(
        self, x: Tensor = None, memory: Tensor = None, src_mask: Tensor = None, trg_mask: Tensor = None, 
        layer_past_self=None, return_present=True):

        if return_present: assert not self.training

        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        h1, present_self = self.trg_trg_att.forward_fast(x_norm, x_norm, x_norm, mask=trg_mask, layer_past=layer_past_self)
        
        h1 = self.dropout(h1) + x

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        h2, _ = self.src_trg_att.forward_fast(memory, memory, h1_norm, mask=src_mask, layer_past=None)
        o = self.feed_forward(self.dropout(h2) + h1)

        return o, present_self


class TransformerDecoder(nn.Module):
    def __init__(
        self, vocab_size,  points_pad, num_layers, num_heads, hidden_size, ff_size, dropout, emb_dropout):
        super(TransformerDecoder, self).__init__()

        # self.max_target_positions = max_target_positions
        self._hidden_size = hidden_size
        self._output_size = vocab_size

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.tag_emb = nn.Parameter(torch.randn(4, 1, hidden_size), requires_grad=True)

        self.point_tok_embedding = WordEmbeddings(embedding_dim=hidden_size//5, vocab_size=vocab_size, 
            pad_idx=points_pad, num_heads=8, norm_type=None, activation_type=None, scale=False, scale_factor=None)

        # self.learn_pe = nn.Embedding(self.max_target_positions + points_pad + 1, 512, points_pad)
        # nn.init.normal_(self.learn_pe.weight, mean=0, std=0.02)
        # nn.init.constant_(self.learn_pe.weight[points_pad], 0)

        self.abs_pe = PositionalEncoding(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.output_layer = nn.Linear(hidden_size//5, self._output_size, bias=False)

        self.register_buffer("window_subsequen_mask", window_subsequent_mask(2200, 20))


    def forward(self, trg_tokens, encoder_output, src_mask, trg_mask, mask_future=True, window_mask_future=True, window_size=None, tag_name=None):
        """x: trg_embed
        """
        # assert trg_mask is not None, "trg_mask required for Transformer"
        # print("trg_tokens: ", trg_tokens.shape)
        if trg_tokens.ndim == 2:
            bsz, tgt_len = trg_tokens.size()
            x = self.point_tok_embedding(trg_tokens, trg_mask)
        elif trg_tokens.ndim == 3:
            # print("the decoder input is embeded!")
            bsz, tgt_len, _ = trg_tokens.size()
            x = trg_tokens
        else:
            raise ValueError("word_token dim is not 2 or 3!")
        
        x = x.view(bsz, tgt_len // 5, 5, -1).view(bsz, tgt_len //5, -1)
        x = x + self.abs_pe(x)
        # print("x: ", x.shape)

        if tag_name is not None:
            if tag_name == "pose":
                x = x + self.tag_emb[0].repeat(bsz, tgt_len//5, 1)
            elif tag_name == "face":
                x = x + self.tag_emb[1].repeat(bsz, tgt_len//5, 1)
            elif tag_name == "rhand":
                x = x + self.tag_emb[2].repeat(bsz, tgt_len//5, 1)
            elif tag_name == "lhand":
                x = x + self.tag_emb[3].repeat(bsz, tgt_len//5, 1)
            else:
                raise ValueError("{} is wrong!".format(tag_name))

        x = self.emb_dropout(x)

        if mask_future:
            if trg_mask is not None:
                trg_mask = trg_mask.unsqueeze(1) & subsequent_mask(x.size(1)).bool().to(x.device)
            else:
                trg_mask = subsequent_mask(x.size(1)).bool().to(x.device)
        
        if window_mask_future:
            assert window_size is not None
            size = x.size(1)
            if trg_mask is not None:
                trg_mask = trg_mask.unsqueeze(1) & self.window_subsequen_mask[:, :size, :size].to(x.device)
            else:
                trg_mask = self.window_subsequen_mask[:, :size, :size].to(x.device)

        for layer in self.layers:
            x = layer(x=x, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask)

        x = self.layer_norm(x)
        x = x.view(bsz, tgt_len//5, 5, -1).contiguous().view(bsz, tgt_len, -1)
        # print("out x: ", x.shape)
        x = self.output_layer(x)
        # print("out x: ", x.shape)
        # exit()
        return x

    def forward_fast(self, trg_tokens, encoder_output, src_mask, trg_mask, mask_future=True, window_mask_future=True, window_size=None, 
                     past_self=None):
        """x: trg_embed
        """
        # inference only
        assert not self.training

        # assert trg_mask is not None, "trg_mask required for Transformer"
        x = self.point_tok_embedding(trg_tokens, trg_mask)

        if past_self is not None:
            past_length = past_self.size(-2)
            assert past_length is not None

            # TODO:
            x = x + self.abs_pe(x, past_length)
        else:
            x = x + self.abs_pe(x)  # add position encoding to word embedding

        x = self.emb_dropout(x)

        presents_self = []  # accumulate over layers
        for i, layer in enumerate(self.layers):
            x, present_self = layer.forward_fast(x=x, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask, 
                layer_past_self=past_self[i, ...] if past_self is not None else None,
                return_present=True)
            
            presents_self.append(present_self)

        x = self.layer_norm(x)
        output = self.output_layer(x)

        return output, torch.stack(presents_self)



if __name__ == "__main__":
    # trg_embed = torch.randn(5, 10, 512)
    # encoder_output = torch.randn(5, 18, 512)
    # src_mask = None
    # trg_mask = None
    # m = TransformerDecoder()
    # o = m(trg_embed, encoder_output, src_mask, trg_mask)
    # print(o.shape)

    # window mask
    size = 12
    window_size = 4
    mask = window_subsequent_mask(size, window_size)
    print(mask, mask.shape)


    
