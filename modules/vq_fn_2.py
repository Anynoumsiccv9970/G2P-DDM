import torch
import torch.nn as nn
import torch.distributed as dist
import torch
from torch.autograd import Function
import torch.nn.functional as F


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                                    inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
                                           index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                   .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


class VQEmbedding(nn.Module):
    def __init__(self, K, D, ema):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)
        self.K = K

        self.ema = ema
        if self.ema:
            self.eps = 1e-5
            self.decay = 0.99
            self.register_buffer('running_size', torch.zeros(K))
            self.register_buffer('running_sum', self.embedding.weight.detach())

    def forward(self, z_e_x):
        bsz, hid, h, t = z_e_x.size()
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        if self.ema:
            # Use EMA to update the embedding vectors
            with torch.no_grad():
                device = indices.device
                size = torch.zeros_like(self.running_size, dtype=indices.dtype, device=device)
                size.index_add_(dim=0, index=indices, source=torch.ones_like(indices, device=device))
                if get_world_size() > 1:
                    size = AllReduce.apply(size)
                self.running_size.data.mul_(self.decay).add_(1 - self.decay, size)

                sum = torch.zeros_like(self.running_sum, dtype=z_e_x_.dtype, device=device)
                b, h, w, c = z_e_x_.size()
                sum.index_add_(dim=0, index=indices, source=z_e_x_.view(b * h * w, c))
                if get_world_size() > 1:
                    sum = AllReduce.apply(sum)
                self.running_sum.data.mul_(self.decay).add_(1 - self.decay, sum)

                n = self.running_size.sum()
                size_ = (self.running_size + self.eps) / (n + self.K * self.eps) * n
                self.embedding.weight.data.copy_(self.running_sum / size_.unsqueeze(1))

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)


        

        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()
        commitment_loss = 0.25 * F.mse_loss(z_e_x, z_q_x_bar.detach())

        return indices.view(bsz, h, t), z_q_x_bar, commitment_loss


class DVQEmbedding(nn.Module):
    def __init__(self, num, K, D, ema):
        super().__init__()
        assert D % num == 0
        self.num = num
        self.D = D
        self.ve = nn.ModuleList([VQEmbedding(K, D // num, ema) for i in range(num)])

    def forward(self, z_e_x):
        assert z_e_x.dim() == 4

        r1 = []
        r2 = []
        commit_loss = 0
        for i, part in enumerate(z_e_x.split(self.D // self.num, dim=1)):
            indices, z_q_x, com_loss = self.ve[i](part)
            r1.append(indices.unsqueeze(1))
            r2.append(z_q_x)
            commit_loss += com_loss
        encoding_indices = torch.cat(r1, dim=1)
        embeddings_st = torch.cat(r2, dim=1)
        commitment_loss = commit_loss
        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss)


if __name__ == "__main__":
    # codebook = VQEmbedding(1024, 256, ema=True)
    # x = torch.randn(5, 256, 10, 10)
    # x = codebook(x)
    # # print(x)
    # print(x[0].shape, x[1].shape)

    codebook = DVQEmbedding(4, 1024, 256, ema=True)
    x = torch.randn(5, 256, 10, 10)
    x = codebook(x)
    # print(x)
    print(x[0].shape, x[1].shape, x[2])
