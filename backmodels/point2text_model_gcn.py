from email.policy import default
from turtle import forward
from matplotlib.pyplot import text
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
from modules.st_gcn import ConvTemporalGraphical, Graph
import torch.distributed as dist
import argparse
from modules.utils import shift_dim
import numpy as np
from data.data_prep.renderopenpose import *
import torchvision
import cv2
from modules.attention import Transformer, FeedForward
from modules.nearby_attn import AttnBlock
from modules.vq_fn import Codebook
import einops
from modules.sp_layer import SPL
from util.plot_videos import draw_frame_2D
# from ctc_decoder import beam_search, best_path
from collections import defaultdict
from util.wer import get_wer_delsubins
import ctcdecode
from itertools import groupby
from util.phoneix_cleanup import clean_phoenix_2014
from util.metrics import wer_single

from models_phoneix.point2text_model_vqvae_tr_nat_stage1_seperate2 import Point2textModel


class BackTranslateModel(pl.LightningModule):
    def __init__(self, args, text_dict):
        super().__init__()

        self.text_dict = text_dict

        # self.points_emb = nn.Linear(150, 512)
        ds_kernels = [2, 2]
        self.pose_gcn = ST_GCN_18(in_channels=3, ds_kernels=ds_kernels, graph_cfg={'layout':'sign_pose', 'strategy':'spatial'})
        self.hand_gcn = ST_GCN_18(in_channels=3, ds_kernels=ds_kernels, graph_cfg={'layout':'hand21', 'strategy':'spatial'}) 

        self.encoder = Encoder(dim=256, depth=3, heads=8, mlp_dim=1024, dropout = 0.1)

        self.ctc_out = nn.Linear(256*3, len(text_dict))

        self.ctcLoss = nn.CTCLoss(text_dict.blank(), reduction="mean", zero_infinity=True)

        self.decoder_vocab = [chr(x) for x in range(20000, 20000 + len(text_dict))]
        self.decoder = ctcdecode.CTCBeamDecoder(self.decoder_vocab, beam_width=5,
                                                blank_id=text_dict.blank(),
                                                num_processes=10)
        self.save_hyperparameters()

    def forward(self, points, skel_len, word_tokens, word_len, mode):
        """[bs, t, 150]
        """
        points = einops.rearrange(points, "b t (v n) -> b n t v", n=3)
        pose = self.pose_gcn(points[..., :8]).mean(-1)
        rhand = self.hand_gcn(points[..., 8:8+21]).mean(-1)
        lhand = self.hand_gcn(points[..., 29:50]).mean(-1)

        skel_len = skel_len // 4 
        max_len = pose.size(-1)
        points_mask = self._get_mask(skel_len, max_len, points.device)
        points_mask = points_mask.unsqueeze_(1).unsqueeze_(1)

        pose = self.encoder(pose, points_mask)
        rhand = self.encoder(rhand, points_mask)
        lhand = self.encoder(lhand, points_mask)

        points = torch.cat([pose, rhand, lhand], dim=-1)
        logits = self.ctc_out(points)  # [bs, t, vocab_size]

        lprobs = logits.log_softmax(-1) # [b t v] 
        lprobs = einops.rearrange(lprobs, "b t v -> t b v")
        
        loss = self.ctcLoss(lprobs.cpu(), word_tokens.cpu(), skel_len.cpu(), word_len.cpu()).to(lprobs.device)
        self.log('{}/loss'.format(mode), loss.detach(), prog_bar=True)
        return loss, logits
        

    def training_step(self, batch):
        word_tokens = batch["gloss_id"]
        word_len = batch["gloss_len"]
        points = batch["skel_3d"]
        skel_len = batch["skel_len"]

        bs, max_len, v = points.size()
        loss, _ = self.forward(points, skel_len, word_tokens, word_len, "train")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        gloss_id = batch["gloss_id"]
        gloss_len = batch["gloss_len"]
        points = batch["skel_3d"]
        skel_len = batch["skel_len"]
        
        _, logits = self.forward(points, skel_len,  gloss_id, gloss_len, "val") # [bs, t, vocab_size]
        bs = skel_len.size(0)
        
        # TODO! recognition prediction and compute wer
        gloss_logits = F.softmax(logits, dim=-1)
        # print("gloss_logits: ", gloss_logits.shape)  # [bs, sgn_len, gloss_vocab_size]
        # print("skel_len: ", skel_len)
        skel_len = skel_len // 4
        pred_seq, _, _, out_seq_len = self.decoder.decode(gloss_logits, skel_len)
        # print("pred_seq: ", pred_seq.shape)        # [bs, reg_beam_size, sgn_len]
        # print("out_seq_len: ", out_seq_len)  # [bs, reg_beam_size]

        err_delsubins = np.zeros([4])
        count = 0
        correct = 0
        total_error = total_del = total_ins = total_sub = total_ref_len = 0
        for i, length in enumerate(gloss_len):
            ref = gloss_id[i][:length].tolist()[:-1]
            hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())][:-1]
            ref_sent = clean_phoenix_2014(self.text_dict.deocde_list(ref))
            hyp_sent = clean_phoenix_2014(self.text_dict.deocde_list(hyp))
            # hyp = ref
            # decoded_dict[vname[i]] = (ref, hyp)
            correct += int(ref == hyp)
            err = get_wer_delsubins(ref, hyp)
            err_delsubins += np.array(err)
            count += 1

            res = wer_single(ref_sent, hyp_sent)
            total_error += res["num_err"]
            total_del += res["num_del"]
            total_ins += res["num_ins"]
            total_sub += res["num_sub"]
            total_ref_len += res["num_ref"]

        return dict(wer=err_delsubins, correct=correct, count=count, 
                    total_error=total_error, total_del=total_del, total_ins=total_ins, total_sub=total_sub, total_ref_len=total_ref_len)

    def validation_epoch_end(self, outputs) -> None:
        val_err, val_correct, val_count = np.zeros([4]), 0, 0
        total_error = total_del = total_ins = total_sub = total_ref_len = 0
        for out in outputs:
            val_err += out["wer"]
            val_correct += out["correct"]
            val_count += out["count"]
            total_error += out["total_error"]
            total_del += out["total_del"]
            total_ins += out["total_ins"]
            total_sub += out["total_sub"]
            total_ref_len += out["total_ref_len"]

        self.log('{}_wer2'.format("val"), total_error / total_ref_len, prog_bar=True)
        self.log('{}/sub2'.format("val"), total_sub / total_ref_len, prog_bar=True)
        self.log('{}/ins2'.format("val"), total_ins / total_ref_len, prog_bar=True)
        self.log('{}/del2'.format("val"), total_del / total_ref_len, prog_bar=True)

        self.log('{}/acc'.format("val"), val_correct / val_count, prog_bar=True)
        self.log('{}_wer'.format("val"), val_err[0] / val_count, prog_bar=True)
        self.log('{}/sub'.format("val"), val_err[1] / val_count, prog_bar=True)
        self.log('{}/ins'.format("val"), val_err[2] / val_count, prog_bar=True)
        self.log('{}/del'.format("val"), val_err[3] / val_count, prog_bar=True)
  
        # for g in self.optimizer.param_groups: 
        #     if self.current_epoch >= 40:           
        #         g['lr'] = g["lr"] * 0.5
        
                # print("Epoch {}, lr {}".format(self.current_epoch, g['lr']))
    
    def _get_mask(self, x_len, size, device):
        pos = torch.arange(0, size).unsqueeze(0).repeat(x_len.size(0), 1).to(device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 3, gamma=0.96, last_epoch=-1)
        return [self.optimizer], [scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        return parser





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

class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = BertLayerNorm(dim)
        self.attn = Attention(heads, dim)
        self.norm2 = BertLayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout = dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x
        

class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(EncoderLayer(dim, heads, mlp_dim, dropout))

    def forward(self, x, mask):
        """x: [bs, h, t]
        """
        x = einops.rearrange(x, "b h t -> b t h")
        for layer in self.layers:
            x = layer(x, mask)
        return x

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
            scores = scores.masked_fill(~mask, float("-inf")) 

        attention = self.softmax(scores)
        context = torch.matmul(attention, v)

        context = (context.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * self.head_size))
        output = self.output_layer(context)
        return output


def zero(x):
    return 0

def iden(x):
    return x
    

class ST_GCN_18(nn.Module):
    r"""Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    def __init__(self,
                 in_channels,
                 ds_kernels,
                 graph_cfg,
                 edge_importance_weighting=True,
                 data_bn=True,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      A.size(1)) if data_bn else iden
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, kernel_size, ds_kernels[0], **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 256, kernel_size, ds_kernels[1], **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, x):

        # data normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(N, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, c, t, v)
        return feature


class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        
        x = self.tcn(x) + res
        x += res

        return self.relu(x), A