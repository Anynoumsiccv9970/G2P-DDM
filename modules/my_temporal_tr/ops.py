# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 01/15/2020
#

import math
from packaging import version
import torch
from .jit_tracing import traceable
from config.options import parse_args


if version.Version(torch.__version__) >= version.Version('1.0.0'):
    from torch import _softmax_backward_data as _softmax_backward_data
else:
    from torch import softmax_backward_data as _softmax_backward_data

__all__ = ['StableDropout', 'MaskedLayerNorm', 'XSoftmax', 'LocalXSoftmax', 'LocalXSoftmaxLocalSpan',
           'SoftLocalXSoftmax']

opts = parse_args()


@traceable
class XSoftmax(torch.autograd.Function):
    """ Masked Softmax which is optimized for saving memory

    Args:

      input (:obj:`torch.tensor`): The input tensor that will apply softmax.
      mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax caculation.
      dim (int): The dimenssion that will apply softmax.

    Example::

      import torch
      from DeBERTa.deberta import XSoftmax
      # Make a tensor
      x = torch.randn([4,20,100])
      # Create a mask
      mask = (x>0).int()
      y = XSoftmax.apply(x, mask, dim=-1)

    """

    @staticmethod
    def forward(self, input, mask, dim):
        """
        """
        # print("="*50)
        # print("4. Xsoftmax:")
        # print("input: ", input.shape)
        # print("mask: ", mask.shape)
        # print("dim: ", dim)
        self.dim = dim
        if version.Version(torch.__version__) >= version.Version('1.2.0a'):
            rmask = ~(mask.bool())
        else:
            rmask = (1 - mask).byte()  # This line is not supported by Onnx tracing.

        output = input.masked_fill(rmask, float('-inf'))

        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        """
        """

        output, = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
        return inputGrad, None, None


@traceable
class LocalXSoftmax(torch.autograd.Function):
    """ Masked Softmax which is optimized for saving memory

    Args:

      input (:obj:`torch.tensor`): The input tensor that will apply softmax.
      mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax caculation.
      dim (int): The dimenssion that will apply softmax.

    Example::

      import torch
      from DeBERTa.deberta import XSoftmax
      # Make a tensor
      x = torch.randn([4,20,100])
      # Create a mask
      mask = (x>0).int()
      y = XSoftmax.apply(x, mask, dim=-1)

    """

    @staticmethod
    def forward(self, input, mask, dim):
        """
        """
        # print("="*50)
        # print("4. Xsoftmax:")
        # print("input: ", input.shape)
        # print("mask: ", mask.shape)
        # print("dim: ", dim)
        self.dim = dim
        if version.Version(torch.__version__) >= version.Version('1.2.0a'):
            rmask = ~(mask.bool())
        else:
            rmask = (1 - mask).byte()  # This line is not supported by Onnx tracing.

        output = input.masked_fill(rmask, float('-inf'))

        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        """
        """
        output, = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
        return inputGrad, None, None


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


@traceable
class SoftLocalXSoftmax(torch.autograd.Function):
    """ Masked Softmax which is optimized for saving memory

    Args:

      input (:obj:`torch.tensor`): The input tensor that will apply softmax.
      mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax caculation.
      dim (int): The dimenssion that will apply softmax.

    Example::

      import torch
      from DeBERTa.deberta import XSoftmax
      # Make a tensor
      x = torch.randn([4,20,100])
      # Create a mask
      mask = (x>0).int()
      y = XSoftmax.apply(x, mask, dim=-1)

    """

    @staticmethod
    def forward(self, input, mask, dim):
        """
        """
        # print("="*50)
        # print("4. Xsoftmax:")
        # print("input: ", input.shape)
        # print("mask: ", mask.shape)
        # print("dim: ", dim)
        self.dim = dim
        if version.Version(torch.__version__) >= version.Version('1.2.0a'):
            rmask = ~(mask.bool())
        else:
            rmask = (1 - mask).byte()  # This line is not supported by Onnx tracing.

        # print("rmask: ", rmask.shape, rmask[0, 0, :20, :20])
        output = input.masked_fill(rmask, float('-inf'))  # [bs, heads, length, length]
        # output = input.masked_fill(rmask, 1e-9)  # [bs, heads, length, length]
        # print("output-1: ", output[0, 0, :20, :20])

        bs, heads, length, _ = output.size()
        WINDOW_SIZE = opts.window_size


        # TODO. soft select.
        distribution = output.softmax(-1)  # [bs, heads, length, length]
        # print("distribution: ", distribution.shape, distribution[0, 0, :30, :30])

        position = torch.arange(0, length).reshape(1, length).repeat(length, 1).unsqueeze_(0).unsqueeze_(1).to(
            distribution.device)  # [bs, heads, length, length]
        position = position.float() * mask.float()
        # print("position: ", position[0, 0, :20, :20])
        mu = torch.sum(distribution * position, dim=-1, keepdim=True)   # [bs, heads, length, 1]
        # print("mu: ", mu[0, 0, :20, :20])

        # local_position = torch.arange(0, 2 * WINDOW_SIZE).reshape(1, 2 * WINDOW_SIZE) + torch.arange(
        #     0, length).reshape(length, 1)
        # local_position = local_position.unsqueeze_(0).unsqueeze_(0).float().to(mu.device)   # [1, 1, length, window_size]

        local_position = torch.arange(0, length).reshape(1, length).repeat(length, 1).to(mu.device)
        local_position = local_position.unsqueeze_(0).unsqueeze_(0).float() * mask.float()  # [1, 1, length, length]
        # local_size = torch.sum(mask.float(), dim=-1, keepdim=True).to(mu.device)  # [bs, 1, length, 1]


        # print("local_position: ", local_position[0, 0, :20, :20])
        # print("local_size: ", local_size[0, 0, :20, :20])
        # print("local_position - mu: ", (local_position - mu)[0, 0, :20, :20])  # [bs, 1, length, length]

        local_position_mu = (local_position - mu) * mask.float()
        # print("local_position - mu: ", local_position_mu[0, 0, :20, :20])  # [bs, 1, length, length]

        # 这里的 (x-mu) 的均值计算有问题
        # skewness = (torch.sum(local_position_mu **3, dim=-1, keepdim=True) / local_size) / (
        #         (torch.sum(local_position_mu **2, dim=-1, keepdim=True)/ local_size) ** (3/2))
        skewness = (torch.sum(local_position_mu **3 * distribution, dim=-1, keepdim=True)) / (
                (torch.sum(local_position_mu **2 * distribution, dim=-1, keepdim=True)) ** (3/2))

        # kurtosis = (torch.sum(local_position_mu ** 4, dim=-1, keepdim=True)/ local_size) / (
        #         (torch.sum(local_position_mu ** 2, dim=-1, keepdim=True) / local_size) ** 2) - 3
        kurtosis = (torch.sum(local_position_mu ** 4 * distribution, dim=-1, keepdim=True)) / (
                (torch.sum(local_position_mu ** 2 * distribution, dim=-1, keepdim=True)) ** 2) - 3
        # print("skewness: ", skewness.shape, skewness[0, 0, :20])
        # print("kurtosis: ", kurtosis.shape, kurtosis[0, 0, :20])

        lambda_r = 1.0
        lrr_right = lambda_r * torch.exp(skewness) * torch.exp(-kurtosis) * WINDOW_SIZE
        lambda_l = 1.0
        lrr_left = lambda_l * torch.exp(-skewness) * torch.exp(-kurtosis) * WINDOW_SIZE

        # print("lrr_left: ", lrr_left[0, 0, :20], lrr_left.shape)
        # print("lrr_right: ", lrr_right[0, 0, :20], lrr_right.shape)

        # local_win = torch.arange(0, length).reshape(1, length).to(mu.device)
        # local_win = local_win.unsqueeze_(0).unsqueeze_(-1).float() # [1, 1, length, 1]

        local_win = mu
        lrr_left = (local_win - lrr_left)
        zero = torch.zeros_like(lrr_left).to(lrr_left.device)
        lrr_left = torch.where(lrr_left > 0, lrr_left, zero)

        lrr_right = (local_win + lrr_right)
        max = (torch.ones_like(lrr_right) * (length-1)).to(lrr_right.device)
        lrr_right = torch.where(lrr_right > (length-1), max, lrr_right)

        # print("local_win: ", local_win[0, 0, :20], local_win.shape)
        # print("lrr_left: ", lrr_left[0, 0, :20], lrr_left.shape)
        # print("lrr_right: ", lrr_right[0, 0, :20], lrr_right.shape)


        # new local_position
        left_mask = (local_position - lrr_left).long()
        # print("left_mask: ", left_mask[0, 0, :20, :20])
        left_mask = (left_mask > 0).long()
        # print("left_mask: ", left_mask[0, 0, :20, :20])

        right_mask = (local_position - lrr_right).long()
        right_mask = (right_mask < 0).long()
        # print("right_mask: ", right_mask[0, 0, :20, :20])


        final_mask = left_mask & right_mask
        # print("final_mask: ", final_mask[0, 0, :20, :20])
        # # exit()

        if version.Version(torch.__version__) >= version.Version('1.2.0a'):
            rfinal_mask = ~(final_mask.bool())
        else:
            rfinal_mask = (1 - final_mask).byte()  # This line is not supported by Onnx tracing.
        # print("final_mask: ", final_mask[0, 0, :20, :20])

        output = output.masked_fill(rfinal_mask, -1e9)
        # print("output-2: ", output[0, 0, :20, :20])

        # TODO. Laplacian kernel
        gamma = 1.0 / (2 * WINDOW_SIZE)
        relative_matrix = build_relative_position(length, length, output.device).unsqueeze_(0)
        # print("relative_matrix: ", relative_matrix.shape, relative_matrix[0, 0, :20, :20])
        lap_kernel = torch.exp(-gamma * torch.abs(relative_matrix).float())
        # print("lap_kernel: ", lap_kernel.shape, lap_kernel[0, 0, :, :])
        output = lap_kernel * output
        # print("output-2: ", output[0, 0, :20, :20])
        # exit()

        # # TODO. hard select
        # WINDOW_SIZE = opts.window_size
        # topk_scores, topk_ids = output.topk(k=WINDOW_SIZE, dim=-1, largest=True)  # [bs, head, q_len, k]
        # threshod = topk_scores[:, :, :, -1].unsqueeze(-1).detach()  # [bs, head, q_len, 1]
        # output = output.masked_fill(output < threshod, -1e9)
        # print("output: ", output.shape, output[0, 0, 0, :])

        output = torch.softmax(output, self.dim)

        # output.masked_fill_(rmask, 0)
        output.masked_fill_(rmask & rfinal_mask, 0)
        # print("output-3: ", output[0, 0, :20, :20])
        # exit()

        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        """
        """

        output, = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
        return inputGrad, None, None


@traceable
class LocalXSoftmaxLocalSpan(torch.autograd.Function):
    """ Masked Softmax which is optimized for saving memory

    Args:

      input (:obj:`torch.tensor`): The input tensor that will apply softmax.
      mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax caculation.
      dim (int): The dimenssion that will apply softmax.

    Example::

      import torch
      from DeBERTa.deberta import XSoftmax
      # Make a tensor
      x = torch.randn([4,20,100])
      # Create a mask
      mask = (x>0).int()
      y = XSoftmax.apply(x, mask, dim=-1)

    """

    @staticmethod
    def forward(self, input, mask, dim, src_length):
        """
        """
        # print("="*50)
        # print("4. Xsoftmax:")
        # print("input: ", input.shape)
        # print("mask: ", mask.shape)
        # print("dim: ", dim)
        self.dim = dim
        if version.Version(torch.__version__) >= version.Version('1.2.0a'):
            rmask = ~(mask.bool())
        else:
            rmask = (1 - mask).byte()  # This line is not supported by Onnx tracing.
        # print(rmask[0, 0, :20, :20])

        output = input.masked_fill(rmask, float('-inf'))
        # print(output.shape, output[0, 0, -20:, -20:])

        # TODO. Soft select. adaptively select the local-span, which is about 80% of the total.
        WINDOW_SIZE = opts.window_size
        distribution = output.softmax(-1)  # [bs, heads, length, length]
        span_local_mask = torch.ones_like(distribution).cuda()
        bs, heads, length, _ = distribution.size()
        for b in range(bs):
            for h in range(heads):
                cue_length = src_length[b]
                for l in range(src_length[b]):
                    percentage = distribution[b, h, l, l]  # current position
                    span_local_mask[b, h, l, l] = 0
                    cur = l
                    cur_left, cur_right = cur - 1, cur + 1
                    while percentage < 0.8:
                        # print("======= cur: ", cur, cur_left, cur_right, percentage)
                        if cur_left < max(0, l - WINDOW_SIZE) and cur_right <= min(l + WINDOW_SIZE, cue_length - 1):
                            percentage += distribution[b, h, l, cur_right]
                            span_local_mask[b, h, l, cur_right] = 0
                            cur_right += 1
                        elif cur_left >= max(0, l - WINDOW_SIZE) and cur_right > min(l + WINDOW_SIZE, cue_length - 1):
                            percentage += distribution[b, h, l, cur_left]
                            span_local_mask[b, h, l, cur_left] = 0
                            cur_left -= 1
                        elif (cur_left >= max(0, l - WINDOW_SIZE) and cur_right <= min(l + WINDOW_SIZE,
                                                                                       cue_length - 1)):
                            if distribution[b, h, l, cur_left] > distribution[b, h, l, cur_right]:
                                percentage += distribution[b, h, l, cur_left]
                                span_local_mask[b, h, l, cur_left] = 0
                                cur_left -= 1
                            else:
                                percentage += distribution[b, h, l, cur_right]
                                span_local_mask[b, h, l, cur_right] = 0
                                cur_right += 1
                        else:
                            raise ValueError("It is impossible!")
        # print("span_local_mask: ", span_local_mask.shape, span_local_mask[0, 0, :20, :20])
        output = output.masked_fill(span_local_mask.byte(), -1e9)
        # print(output.shape, output[0, 0, -20:, -20:])
        # exit()
        # TODO, hard select using topk function
        WINDOW_SIZE = opts.window_size
        topk_scores, topk_ids = output.topk(k=WINDOW_SIZE, dim=-1, largest=True)  # [bs, head, q_len, k]
        threshod = topk_scores[:, :, :, -1].unsqueeze(-1).detach()  # [bs, head, q_len, 1]
        output = output.masked_fill(output < threshod, -1e9)

        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        """
        """

        output, = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
        return inputGrad, None, None, None


class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        if version.Version(torch.__version__) >= version.Version('1.2.0a'):
            mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()
        else:
            mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).byte()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


@traceable
class XDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            mask, = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


class StableDropout(torch.nn.Module):
    """ Optimized dropout module for stabilizing the training

    Args:

      drop_prob (float): the dropout probabilities

    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """ Call the module

        Args:

          x (:obj:`torch.tensor`): The input tensor to apply dropout


        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


def MaskedLayerNorm(layerNorm, input, mask=None):
    """ Masked LayerNorm which will apply mask over the output of LayerNorm to avoid inaccurate updatings to the LayerNorm module.

    Args:
      layernorm (:obj:`~DeBERTa.deberta.BertLayerNorm`): LayerNorm module or function
      input (:obj:`torch.tensor`): The input tensor
      mask (:obj:`torch.IntTensor`): The mask to applied on the output of LayerNorm where `0` indicate the output of that element will be ignored, i.e. set to `0`

    Example::

      # Create a tensor b x n x d
      x = torch.randn([1,10,100])
      m = torch.tensor([[1,1,1,0,0,0,0,0,0,0]], dtype=torch.int)
      LayerNorm = DeBERTa.deberta.BertLayerNorm(100)
      y = MaskedLayerNorm(LayerNorm, x, m)

    """
    output = layerNorm(input).to(input)
    if mask is None:
        return output
    if mask.dim() != input.dim():
        if mask.dim() == 4:
            mask = mask.squeeze(1).squeeze(1)
        mask = mask.unsqueeze(2)
    mask = mask.to(output.dtype)
    return output * mask
