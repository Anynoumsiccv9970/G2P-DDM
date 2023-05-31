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
from ctc_decoder import beam_search, best_path
from collections import defaultdict
from util.wer import get_wer_delsubins
import ctcdecode
from itertools import groupby
from util.phoneix_cleanup import clean_phoenix_2014
from util.metrics import wer_single


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.bn1 = self.norm_layer(planes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # downsample
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = self.norm_layer(planes, affine=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MainStream(nn.Module):
    def __init__(self, vocab_size, momentum=0.1):
        super(MainStream, self).__init__()

        # cnn
        # first layer: channel 3 -> 32
        self.conv = conv3x3(3, 32)
        self.bn = nn.BatchNorm2d(32, momentum=momentum,  affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        channels = [32, 64, 128, 256, 512]
        layers = []
        for num_layer in range(len(channels) - 1):
            layers.append(BasicBlock(channels[num_layer], channels[num_layer + 1]))
        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)

        # encoder G1, two F5-S1-P2-M2
        self.enc1_conv1 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.enc1_bn1 = nn.BatchNorm1d(512, momentum=momentum, affine=True)
        self.enc1_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc1_conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.enc1_bn2 = nn.BatchNorm1d(512, momentum=momentum, affine=True)
        self.enc1_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.enc1 = nn.Sequential(self.enc1_conv1, self.enc1_bn1, self.relu, self.enc1_pool1, self.enc1_conv2, self.enc1_bn2, self.relu, self.enc1_pool2)

        self.enc2_conv = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.enc2_bn = nn.BatchNorm1d(1024, momentum=momentum, affine=True)
        self.enc2 = nn.Sequential(self.enc2_conv, self.enc2_bn, self.relu)

        self.fc = nn.Linear(1024, vocab_size)

        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear)):
                m.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()


    def forward(self, video, len_video=None):
        """
        x: [batch, num_f, 3, h, w]
        """
        # print("input: ", video.size())
        bs, num_f, c, h, w = video.size()

        x = video.reshape(-1, c, h, w)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.layers(x)
        x = self.avgpool(x).squeeze_()  # [bs*t, 512]

        x = x.reshape(bs, -1, 512)     # [bs, t ,512]
        x = x.permute(0, 2, 1)  # [bs, 512, t]

        # enc1
        x = self.enc1_conv1(x)  # [bs, 512, t/2]
        x = self.enc1_bn1(x)
        x = self.relu(x)
        x = self.enc1_pool1(x)  # [bs, 512, t/2]

        x = self.enc1_conv2(x)  # [bs, 512, t/2]
        x = self.enc1_bn2(x)
        x = self.relu(x)
        x = self.enc1_pool2(x)  # [bs, 512, t/4]

        # enc2
        x = self.enc2_conv(x)
        x = self.enc2_bn(x)
        out = self.relu(x)

        out = F.dropout(out, p=0.4, training=self.training)

        out = out.permute(0, 2, 1).contiguous()
        
        logits = self.fc(out)  # [batch, t/4, vocab_size]
        return logits


class BackTranslateModel(pl.LightningModule):
    def __init__(self, args, text_dict):
        super().__init__()

        self.text_dict = text_dict

        self.cnn_model = MainStream(len(text_dict))

        self.ctcLoss = nn.CTCLoss(text_dict.blank(), reduction="mean", zero_infinity=True)

        self.decoder_vocab = [chr(x) for x in range(20000, 20000 + len(text_dict))]
        self.decoder = ctcdecode.CTCBeamDecoder(self.decoder_vocab, beam_width=5,
                                                blank_id=text_dict.blank(),
                                                num_processes=4)

        self.save_hyperparameters()


    def forward(self, video, skel_len, word_tokens, word_len, mode):
        """[bs, num_frame, 2, 112, 112]
        """
        logits = self.cnn_model(video)
        skel_len = skel_len // 4 

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
        video = batch["vid"]

        loss, _ = self.forward(video, skel_len,  word_tokens, word_len, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        gloss_id = batch["gloss_id"]
        gloss_len = batch["gloss_len"]
        points = batch["skel_3d"]
        skel_len = batch["skel_len"]
        video = batch["vid"]

        _, logits = self.forward(video, skel_len,  gloss_id, gloss_len, "val") # [bs, t, vocab_size]
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

            # res = wer_single(ref_sent, hyp_sent)
            # total_error += res["num_err"]
            # total_del += res["num_del"]
            # total_ins += res["num_ins"]
            # total_sub += res["num_sub"]
            # total_ref_len += res["num_ref"]

        return dict(wer=err_delsubins, correct=correct, count=count)
                    # total_error=total_error, total_del=total_del, total_ins=total_ins, total_sub=total_sub, total_ref_len=total_ref_len)

    def validation_epoch_end(self, outputs) -> None:
        val_err, val_correct, val_count = np.zeros([4]), 0, 0
        total_error = total_del = total_ins = total_sub = total_ref_len = 0
        for out in outputs:
            val_err += out["wer"]
            val_correct += out["correct"]
            val_count += out["count"]
            # total_error += out["total_error"]
            # total_del += out["total_del"]
            # total_ins += out["total_ins"]
            # total_sub += out["total_sub"]
            # total_ref_len += out["total_ref_len"]

        # self.log('{}_wer2'.format("val"), total_error / total_ref_len, prog_bar=True)
        # self.log('{}/sub2'.format("val"), total_sub / total_ref_len, prog_bar=True)
        # self.log('{}/ins2'.format("val"), total_ins / total_ref_len, prog_bar=True)
        # self.log('{}/del2'.format("val"), total_del / total_ref_len, prog_bar=True)

        self.log('{}/acc'.format("val"), val_correct / val_count, prog_bar=True)
        self.log('{}_wer'.format("val"), val_err[0] / val_count, prog_bar=True)
        # self.log('{}/sub'.format("val"), val_err[1] / val_count, prog_bar=True)
        # self.log('{}/ins'.format("val"), val_err[2] / val_count, prog_bar=True)
        # self.log('{}/del'.format("val"), val_err[3] / val_count, prog_bar=True)

        print("wer and acc: ", val_correct / val_count, val_err[0] / val_count)
        # for g in self.optimizer.param_groups: 
        #     if self.current_epoch >= 40:           
        #         g['lr'] = g["lr"] * 0.5
        
        #         print("Epoch {}, lr {}".format(self.current_epoch, g['lr']))
    
    def _get_mask(self, x_len, size, device):
        pos = torch.arange(0, size).unsqueeze(0).repeat(x_len.size(0), 1).to(device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 3, gamma=0.96, last_epoch=-1)
        return [self.optimizer]


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