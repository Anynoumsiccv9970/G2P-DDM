# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 01/15/2020
#

"""
  Disentangled SelfAttention module
"""

import torch
import math
from .ops import *
import random

__all__ = ['build_relative_position', 'DisentangledLocalSelfAttention']


def out_window_mask(size, window_size=4, shift_size=0):
    """
    :param size:
    :param local_ws:
    :return:
    if not right_shift:
    tensor([[
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]])
    if right_shift:
    tensor([[
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]])
    """
    # print("size, window_size, shift_size: ", size, window_size, shift_size)
    cur = (torch.arange(0, size) - shift_size) % window_size
    left = torch.clamp(torch.arange(0, size) - cur, 0, size)
    right = torch.clamp(torch.arange(0, size) + (window_size - cur), 0, size)
    pos = torch.arange(0, size).unsqueeze(1).repeat(1, size)
    mask = (pos >= left) & (pos < right)
    return mask.unsqueeze(0)  # [1, length, length]


def build_relative_position(query_size, key_size, device):
    """ Build relative position according to the query and key

    We assume the absolute position of query :math:`P_q` is range from (0, query_size) and the absolute position of key :math:`P_k` is range from (0, key_size),
    The relative positions from query to key is

    :math:`R_{q \\rightarrow k} = P_q - P_k`

    Args:
        query_size (int): the length of query
        key_size (int): the length of key

    Return:
        :obj:`torch.LongTensor`: A tensor with shape [1, query_size, key_size]

    """

    q_ids = torch.arange(query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


# @torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    # [bs, head, length, length]
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


# @torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


# @torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


class DisentangledLocalSelfAttention(torch.nn.Module):
    """ Disentangled self-attention module

    Parameters:
        config (:obj:`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to `BertConfig`, \
            for more details, please refer :class:`~DeBERTa.deberta.ModelConfig`

    """

    def __init__(self, config, hidden_size, num_heads, window_size, spatial_attn=False):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_heads))
        self.config = config
        self.window_size = window_size
        self.num_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size
        self.in_proj = torch.nn.Linear(hidden_size, self.all_head_size * 3, bias=False)
        self.q_bias = torch.nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        self.v_bias = torch.nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        if not spatial_attn:
            self.pos_att_type = [x.strip() for x in getattr(config, 'pos_att_type', 'none').lower().split('|')]  # c2p|p2c
            print("temporal self-attention, with c2p and p2c")
        else:
            self.pos_att_type = ["none"]
            print("spatial self-attention, no p2c and c2p")

        self.relative_attention = getattr(config, 'relative_attention', False)
        self.talking_head = getattr(config, 'talking_head', False)

        if self.talking_head:
            self.head_logits_proj = torch.nn.Linear(num_heads, num_heads, bias=False)
            self.head_weights_proj = torch.nn.Linear(num_heads, num_heads, bias=False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = StableDropout(config.dropout)

            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_proj = torch.nn.Linear(hidden_size, self.all_head_size, bias=False)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_q_proj = torch.nn.Linear(hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None,
                rel_embeddings=None):
        """  Call the module

        Args:
            hidden_states (:obj:`torch.FloatTensor`):
                Input states to the module usally the output from previous layer, it will be the Q,K and V in `Attention(Q,K,V)`

            attention_mask (:obj:`torch.ByteTensor`):
                An attention mask matrix of shape [`B`, `N`, `N`] where `B` is the batch size, `N` is the maxium sequence length in which element [i,j] = `1` means the `i` th token in the input can attend to the `j` th token.

            return_att (:obj:`bool`, optional):
                Whether return the attention maxitrix.

            query_states (:obj:`torch.FloatTensor`, optional):
                The `Q` state in `Attention(Q,K,V)`.

            relative_pos (:obj:`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [`B`, `N`, `N`] with values ranging in [`-max_relative_positions`, `max_relative_positions`].

            rel_embeddings (:obj:`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [:math:`2 \\times \\text{max_relative_positions}`, `hidden_size`].


        """
        if query_states is None:
            qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
            query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
        else:
            def linear(w, b, x):
                if b is not None:
                    return torch.matmul(x, w.t()) + b.t()
                else:
                    return torch.matmul(x, w.t())  # + b.t()

            ws = self.in_proj.weight.chunk(self.num_heads * 3, dim=0)
            qkvw = [torch.cat([ws[i * 3 + k] for i in range(self.num_heads)], dim=0) for k in range(3)]
            qkvb = [None] * 3

            q = linear(qkvw[0], qkvb[0], query_states)
            k, v = [linear(qkvw[i], qkvb[i], hidden_states) for i in range(1, 3)]
            query_layer, key_layer, value_layer = [self.transpose_for_scores(x) for x in [q, k, v]]

        # print("="*50)
        # print("1. Projection:")

        query_layer += self.transpose_for_scores(self.q_bias.unsqueeze(0).unsqueeze(0))
        value_layer += self.transpose_for_scores(self.v_bias.unsqueeze(0).unsqueeze(0))

        # print("query_layer: ", query_layer.shape)
        # print("key_layer: ", key_layer.shape)
        # print("value_layer: ", value_layer.shape)
        
        # print("=" * 50)
        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1
        scale = math.sqrt(query_layer.size(-1) * scale_factor)
        query_layer = query_layer / scale

        # print("=" * 50)
        # print("2. content to content: ")
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # print("attention_scores: ", attention_scores.shape)
        # print("=" * 50)

        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

        if rel_att is not None:
            attention_scores = (attention_scores + rel_att)

        # print("attention_scores: ", attention_scores.shape)
        # bxhxlxd
        if self.talking_head:
            attention_scores = self.head_logits_proj(attention_scores.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # TODO? shift_size
        if self.training:
            shift_size = random.randint(0, self.window_size)
        else:
            # ensamble
            shift_size = random.randint(0, self.window_size)
        win_mask = out_window_mask(
            size=attention_scores.size(2),
            window_size=self.window_size,
            shift_size=shift_size).unsqueeze(1).to(attention_scores.device)
        # print("win_mask: ", win_mask.shape)

        # print("attention_mask: ", attention_mask.shape, attention_mask[0])
        # print("attention_scores: ", attention_scores.shape)
        if attention_mask is not None:
            attention_mask = attention_mask & win_mask.byte()
        else:
            attention_mask = win_mask.byte()
        # print("attention_mask: ", attention_mask.shape, attention_mask[0, :, 0, :])
        
        attention_probs = LocalXSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        if self.talking_head:
            attention_probs = self.head_weights_proj(attention_probs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_att:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        # print("="*50)
        # print("3. content to position: ")
        if relative_pos is None:
            q = query_layer.size(-2)
            # print("="*20 + "3.1. build relative distance matrix: ")
            relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device)
            # print("relative_pos: ", relative_pos.shape)
            # print("relative_pos: ", relative_pos)

        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim() != 4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')

        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        # print("att_span: ", att_span)
        relative_pos = relative_pos.long().to(query_layer.device)
        # print("rel_embeddings: ", rel_embeddings.shape)
        rel_embeddings = rel_embeddings[self.max_relative_positions - att_span:self.max_relative_positions + att_span,
                         :].unsqueeze(0)
        # TODO, relative_position_embedding: [1, 2*max_relative_position, hidden_size]
        # print("rel_embeddings: ", rel_embeddings.shape)

        # print("=" * 20 + "3.2. content to position: ")

        score = 0
        # content->position
        if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            # print("pos_key_layer: ", pos_key_layer.shape)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)
            # TODO: pos_key_layer: [1, head, 2*max_relative_position, hidden_size]
            # print("pos_key_layer: ", pos_key_layer.shape)

        if 'c2p' in self.pos_att_type:
            # [bs, head, length, head_size] x [1, head, 2*max_relative_position, hidden_size]
            # -> [bs, head, length, 2*max_relative_position]
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            # print("c2p_att: ", c2p_att.shape)
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            # print("c2p_pos: ", c2p_pos.shape, c2p_pos)
            # print("c2p_pos index: ", c2p_dynamic_expand(c2p_pos, query_layer, relative_pos).shape)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            # print("c2p_att: ", c2p_att.shape)
            score += c2p_att

        # position->content
        # print("=" * 20 + "3.3. position to content: ")

        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            # print("pos_query_layer: ", pos_query_layer.shape)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)
            # TODO: pos_query_layer: [1, head, 2*max_relative_position, hidden_size]
            # print("pos_query_layer: ", pos_query_layer.shape)

        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1) * scale_factor)
            if query_layer.size(-2) != key_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            else:
                r_pos = relative_pos
            # print("r_pos: ", r_pos)

            # TODO: note the difference between c2p_pos (line:251)
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            # print("p2c_pos: ", p2c_pos)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if 'p2c' in self.pos_att_type:
            # [bs, head, length, head_size] x [1, head, 2*max_relative_position, head_size]
            # -> [bs, head, length, 2*max_relative_position]
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            # print("p2c_att: ", p2c_att.shape)
            p2c_att = torch.gather(p2c_att, dim=-1,
                                   index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)).transpose(-1, -2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_dynamic_expand(pos_index, p2c_att, key_layer))
            score += p2c_att
        # print("score: ", score.shape)
        return score


if __name__ == "__main__":
    import torch.nn as nn
    def test_get_rel_pos(hidden_states, query_states=None, relative_pos=None):
        if relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), hidden_states.device)
        return relative_pos


    batch, length, hid_size = 2, 10, 512


    class Config():
        hidden_size = 512
        num_heads = 8
        relative_attention = True
        max_relative_positions = 3
        max_position_embeddings = 512
        dropout = 0.1
        pos_att_type = "c2p|p2c"
        talking_head = True
        window_size = 4

    config = Config()

    de_sa = DisentangledLocalSelfAttention(config, 16, shift_size=4)

    hidden_states = torch.randn(batch, length, hid_size)
    return_att = False
    query_states = None

    # attention mask
    def get_attention_mask(attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # print("extended_attention_mask: ", extended_attention_mask.shape)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.byte()
            # print("attention_mask: ", attention_mask.shape, attention_mask)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask
    attention_mask = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,0, 0, 0]])
    attention_mask = get_attention_mask(attention_mask.ne(0))


    relative_attention = getattr(config, 'relative_attention', False)
    if relative_attention:
        max_relative_positions = getattr(config, 'max_relative_positions', -1)
        if max_relative_positions < 1:
            max_relative_positions = config.max_position_embeddings
        rel_embeddings = nn.Embedding(max_relative_positions * 2, hidden_size)
    else:
        rel_embeddings = None

    rel_embeddings = rel_embeddings.weight

    relative_pos = None

    def get_rel_pos(hidden_states, query_states=None, relative_pos=None):
        if relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), hidden_states.device)
        return relative_pos

    relative_pos = get_rel_pos(hidden_states, query_states, relative_pos)


    out = de_sa(hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None,
                rel_embeddings=rel_embeddings)
    print("out: ", out.shape)