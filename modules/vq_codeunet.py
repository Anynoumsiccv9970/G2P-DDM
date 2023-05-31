
from turtle import forward
import torch, math
import torch.nn as nn
import torch.nn.functional as F



class CodeUnet(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, diffusion_step):
        super().__init__()
        self.unet_enc = UnetEncoder(dim, depth, heads, mlp_dim, dropout, diffusion_step)
        self.unt_dec = UnetDecoder(dim, depth, heads, mlp_dim, dropout, diffusion_step)
    
    def forward(self, x, x_mask, enc_out, src_mask, t):
        x, mid_states = self.unet_enc(x, x_mask, True)
        x = self.unt_dec(x, x_mask, enc_out, src_mask, t, mid_states)
        return x



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, num_heads, size):
        super(Attention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v,mask):
        batch_size = k.size(0)
        num_heads = self.num_heads

        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2) # [bs, head, length, hid_size]

        # compute scores
        q = q / math.sqrt(self.head_size)
        scores = torch.matmul(q, k.transpose(2, 3)) # [bs, head, q_len, kv_len]

        if mask is not None:
            # print("scores: ", scores.shape, mask.shape)
            scores = scores.masked_fill(~mask, float("-inf")) 

        attention = self.softmax(scores)
        context = torch.matmul(attention, v)

        context = (context.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * self.head_size))
        output = self.output_layer(context)
        return output

class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, ):
        super().__init__()

        self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x

class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, diffusion_step):
        super().__init__()
        
        self.self_attn = Attention(heads, dim)
        self.cross_attn = Attention(heads, dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout = dropout)

        self.norm1 = AdaLayerNorm(dim, diffusion_step)
        self.norm2  = AdaLayerNorm(dim, diffusion_step)
        self.norm3  = AdaLayerNorm(dim, diffusion_step)

        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x, enc_out, tgt_mask, src_mask, t):
        residual = x
        x = self.norm1(x, t)
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = residual + x
        
        residual = x
        x = self.norm2(x, t)
        x = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm3(x, t)
        x = self.ffn(x)
        x = residual + x
        return x

class UnetDecoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0., diffusion_step=100):
        super().__init__()
        self.increase_list = [2,4]
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(DecoderLayer(dim, heads, mlp_dim, dropout, diffusion_step))

    def forward(self, x, pad_future_mask, enc_out, src_mask, t, mid_states):
        bs, l, d = x.size()
        cur_mask = pad_future_mask[:, :, :, 1::4]
        for i, layer in enumerate(self.layers):
            if i in self.increase_list:
                if i == 2:
                    cur_mask = pad_future_mask[:, :, :, 1::2]
                else:
                    cur_mask = pad_future_mask
                x = torch.cat([x.unsqueeze(2)]*2, dim=2) # [bs, t, 2, size] -> [bs, t*2, size]
                l *= 2
                x = x.reshape(bs, l, d)
                x += mid_states.pop(-1)
                x = layer(x, enc_out, cur_mask, src_mask, t)
            else:
                x += mid_states.pop(-1)
                x = layer(x, enc_out, cur_mask, src_mask, t)
        return x

class UnetEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0., diffusion_step=100):
        super().__init__()
        self.decrease_list = [2,4]
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(EncoderLayer(dim, heads, mlp_dim, dropout, diffusion_step))

    def forward(self, x, mask, return_inter=False):
        """x: [bs, t, size]
        """
        mid_states = []
        for i, layer in enumerate(self.layers):
            if i in self.decrease_list:
                q = x[:, 1::2, :]
                kv = x
                x = layer(q, kv, mask)
                mask = mask[:, :, :, 1::2]
            else:
                x = layer(x, x, mask)
            if return_inter: mid_states.append(x)
        if return_inter:
            return x, mid_states
        else:
            return x


class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, diffusion_step):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(heads, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, mask):
        residual = q
        q = self.norm1(q)
        q = self.attn(q, kv, kv, mask)
        q = self.dropout(q)
        x = residual + q

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x

if __name__ == "__main__":
    x = torch.randn(5, 12, 512)
    # enc = UnetEncoder(512, 6, 8, 2048, 0.1)
    x_mask = torch.ones(5, 1, 1, 12).bool()
    # out, mid_states = enc(x, mask, True)
    # print(out.shape)
    # # print("="*10)
    # # for m in mid_states:
    # #     print(m.shape)
    # # print("="*10)

    # dec = UnetDecoder(512, 6, 8, 2048, 0.1)
    enc_out = torch.randn(5, 4, 512)
    src_mask = torch.ones(5, 1, 1, 4).bool()
    t = torch.full((5,), 4)

    m = CodeUnet(512, 6, 8, 2048, 0.1, 100)
    out = m(x, x_mask, enc_out, src_mask, t)
    print(out.shape)
    # out = dec(out, pad_future_mask, enc_out, src_mask, t, mid_states)
    # print(out.shape)

    