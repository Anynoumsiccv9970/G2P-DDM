from email.policy import default
from statistics import mode
from matplotlib.pyplot import axis, text
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
from modules.utils import shift_dim
import numpy as np
from data.data_prep.renderopenpose import *
import torchvision
import cv2
from modules.vq_fn import Codebook
import einops
from modules.sp_layer import SPL
from util.plot_videos import draw_frame_2D
from util.wer import get_wer_delsubins
import ctcdecode
from itertools import groupby
from modules.mask_strategy import *
from util.dtw import calculate_dtw, dtw
import torchvision.transforms as transforms
from data.vocabulary import Dictionary
from util.train_utils import instantiate_from_config
import time
from util.phoneix_cleanup import clean_phoenix_2014
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from modules.vq_codeunet import CodeUnet


NOW_TIME = "-".join(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())[5:-3].split(":")).replace(" ", "-")

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30)) # 0 -> 1e-30
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def alpha_schedule(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:]/att[:-1]
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct
    bt = (1-at-ct)/N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1-att-ctt)/N
    return at, bt, ct, att, btt, ctt

class Point2textModelStage2(pl.LightningModule):
    def __init__(self,
        vocab_file,
        stage1_model_config,
        backtrans_model_config,
        sample_dir,
        learning_rate,
        hidden_size,
        depth,
        heads,
        dropout,
        use_discrete_cfg=True,
        empty_text_prob=0.1,
        unconditional_guidance_scale=5.0,
        resume_ckpt="",
        diffusion_step=100,
        lr_schedule=None,
        optim_type="adam",
        alpha_init_type='alpha1',
        auxiliary_loss_weight=0,
        adaptive_auxiliary_loss=False,
        mask_weight=[1,1]):
        super().__init__()

        self.text_dict = Dictionary.load(vocab_file)
        self.sample_dir = sample_dir
        self.learning_rate = learning_rate
        self.resume_ckpt = resume_ckpt
        self.optim_type = optim_type
        self.lr_schedule = lr_schedule
        self.use_discrete_cfg = use_discrete_cfg
        self.empty_text_prob = empty_text_prob
        self.unconditional_guidance_scale = unconditional_guidance_scale

        self.num_timesteps = diffusion_step
        self.parametrization = 'x0'
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight

        # vqvae
        pose_vqvae = instantiate_from_config(stage1_model_config, init_model=False)
        if not os.path.exists(stage1_model_config.ckpt_path):
            raise ValueError("{} is not existed!".format(stage1_model_config.ckpt_path))
        else:
            print("=== load vqvae model from {}".format(stage1_model_config.ckpt_path))
            self.vqvae =  pose_vqvae.load_from_checkpoint(stage1_model_config.ckpt_path, strict=False)
            if not hasattr(self.vqvae, "codebook"):
                self.vqvae.concat_codebook()
        
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        back_translate_model1 = instantiate_from_config(backtrans_model_config, init_model=False)
        if not os.path.exists(backtrans_model_config.ckpt_path):
            raise ValueError("{} is not existed!".format(backtrans_model_config.ckpt_path))
        else:
            print("=== load back-translate model from {}".format(backtrans_model_config.ckpt_path))
            self.back_translate_model1 =  back_translate_model1.load_from_checkpoint(backtrans_model_config.ckpt_path)
        for p in self.back_translate_model1.parameters():
            p.requires_grad = False
        self.back_translate_model1.eval()

        # encoder-decoder
        self.gloss_embedding = nn.Embedding(len(self.text_dict), hidden_size, self.text_dict.pad())
        self.gloss_embedding.weight.data.normal_(mean=0.0, std=0.02)
        
        stage1_ncodes = self.vqvae.codebook.embeddings.size(0)
        self.pad_id = stage1_ncodes # 2048
        self.mask_id = stage1_ncodes + 1 # 2049
        self.points_vocab_size = stage1_ncodes + 2
        self.point_embedding = nn.Embedding(self.points_vocab_size, hidden_size, self.pad_id)
        self.point_embedding.weight.data.normal_(mean=0.0, std=0.02)

        self.num_classes = self.points_vocab_size
        # self.tem_pos_emb = nn.Parameter(torch.zeros(2000, 512))
        # self.tem_pos_emb.data.normal_(0, 0.02)
        self.tem_pos_emb = PositionalEncoding(0.1, hidden_size, 2000)

        self.encoder = Encoder(dim=hidden_size, depth=depth, heads=heads, mlp_dim=hidden_size*4, dropout=dropout, diffusion_step=diffusion_step)
        self.decoder = CodeUnet(dim=hidden_size, depth=depth, heads=heads, mlp_dim=hidden_size*4, dropout=dropout, diffusion_step=diffusion_step)
        self.out_layer = nn.Linear(hidden_size, self.points_vocab_size-2) 

        self.loss_type = 'vb_stochastic'

        if alpha_init_type == "alpha1":
            at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_classes-1)
        else:
            print("alpha_init_type is Wrong !! ")

        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
        self.zero_vector = None

        self.random = np.random.RandomState(1234)
        self.resume()
        self.save_hyperparameters()
        # exit()
        

    def resume(self, ):
        if os.path.exists(self.resume_ckpt):
            print("=== Load from {}!".format(self.resume_ckpt))
            state_dict = torch.load(self.resume_ckpt)["state_dict"]
            self.load_state_dict(state_dict, strict=True)
            del state_dict
            # self.load_from_checkpoint(self.resume_ckpt, strict=True)
        else:
            print("=== {} is not existed, training from scratch".format(self.resume_ckpt))

    def get_cond_emb(self, gloss_id, gloss_len, repeat_len, skel_len, tgt_mask):
        """vq_tokens: [bs, t, 3]
        """
        
        bs = gloss_id.size(0)
        src_emb = self.gloss_embedding(gloss_id)
        src_emb = self.tem_pos_emb(src_emb)
        src_mask = gloss_id.ne(self.text_dict.pad())
        enc_feat = self.encoder(src_emb, src_mask)

        tgt_inp = torch.zeros(bs, max(skel_len), enc_feat.size(-1)).to(gloss_id.device)
        for i in range(bs):
            start = 0
            for j in range(gloss_len[i].item()):
                end = start + repeat_len[i][j]
                tgt_inp[i, start:end] = enc_feat[i, j]
                start = end
        tgt_inp = einops.repeat(tgt_inp, "b t h -> b (t n) h", n=3)
        return tgt_inp
    
    def cond_decoder(self, x_t, cond_emb, t, x_mask, src_mask):
        """x_t: [bs, t, n]
           cond_emb: [bs, t, emb_dim]
        """
        x_t_emb = self.point_embedding(x_t)
        x_t_emb = self.tem_pos_emb(x_t_emb)
        out_feat = self.decoder(x_t_emb, x_mask.unsqueeze(1).unsqueeze(2), cond_emb, src_mask.unsqueeze(1).unsqueeze(2), t)
        logits = self.out_layer(out_feat) 
        out = einops.rearrange(logits, 'b l c -> b c l')
        return out


    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all(): # 当所有的timestep次数都超过10时，就不使用 uniform 了
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001 # Lt_history 时 KL_loss**2 的滑动平均
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum() # 根据KL_loss的比例来分配timestep被采样的概率

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError
    
    def q_pred(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        # log_probs = torch.cat(
        #     [   log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt),
        #         log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct)
        #     ],
        #     dim=1
        # )
        log_probs = torch.cat(
            [   log_add_exp(log_x_start[:,:-2,:]+log_cumprod_at, log_cumprod_bt), # 前N-2个词
                torch.log(torch.zeros_like(log_x_start[:, -1:, :]).fill_(1e-30)), # 第N个词，也就是pad,
                log_add_exp(log_x_start[:,-2:-1,:]+log_1_min_cumprod_ct, log_cumprod_ct), # 第N-1个词，也就是mask,
                
            ],
            dim=1
        )
        return log_probs

    

    def log_sample_categorical(self, logits, x_mask):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        sample = sample.masked_fill(~x_mask, self.pad_id)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t, x_mask):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0, x_mask)
        return log_sample

    def predict_start(self, log_x_t, cond_emb, t, x_mask, src_mask):          # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        
        out = self.cond_decoder(x_t, cond_emb, t, x_mask, src_mask)

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes - 2
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        # if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
        self.zero_vector = torch.zeros(batch_size, 2, x_t.size(-1)).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)
        return log_pred

    def q_pred_one_timestep(self, log_x_t, t):         # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:-2,:]+log_at, log_bt),
                torch.log(torch.zeros_like(log_x_t[:, -1:, :]).fill_(1e-30)), # 第N个词，也就是pad,
                log_add_exp(log_x_t[:, -2:-1, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )

        return log_probs

    def q_posterior(self, log_x_start, log_x_t, t):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.mask_id).unsqueeze(1) 
        log_one_vector = torch.zeros(batch_size, 2, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, log_x_t.size(-1))

        log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
        # print("log_qt: ", log_qt.shape)
        
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:,:-2,:]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        # print("log_cumprod_ct: ", log_cumprod_ct.shape)

        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-2, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector
        # print("log_qt: ", log_qt.shape)
        

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-2,:], log_zero_vector), dim=1)
        # print("log_qt_one_timestep: ", log_qt_one_timestep.shape)
        

        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_classes-2, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        
        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:,:-2,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        # print("log_EV_xtmin_given_xt_given_xstart: ", log_EV_xtmin_given_xt_given_xstart.shape)
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def _train_loss(self, x, x_mask, cond_emb, src_mask, mode):                       # get the KL loss
        """x: [bs, t]
           cond_emb: [bs, t, emb_dim]
        """
        b, device = x.size(0), x.device
        assert self.loss_type == 'vb_stochastic'
        x_start = x
        # print("length: ", (x != self.pad_id).sum(-1))
        # print("x_start: ", x_start.shape, x_start)
        t, pt = self.sample_time(b, device, 'importance')
        # print("t: ", t, pt)

        log_x_start = index_to_log_onehot(x_start, self.num_classes) # [bs, v, t]
        log_xt = self.q_sample(log_x_start=log_x_start, t=t, x_mask=x_mask)
        xt = log_onehot_to_index(log_xt)
        # print("masked token: ", (xt == self.mask_id).sum(-1))
        # print("xt: ", xt, xt.shape, log_xt.shape)

         ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(log_xt, cond_emb, t=t, x_mask=x_mask, src_mask=src_mask)            # P_theta(x0|xt)
        # print("log_x0_recon: ", log_x0_recon.shape, log_xt.shape)
        
        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)      # go through q(xt_1|xt,\tilde x0)
        # print("log_model_prob: ", log_model_prob.shape)
        
        # compute log_true_prob now 
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t) # go through q(xt_1|xt,\tilde x0)
        # print("log_true_prob: ", log_true_prob.shape)

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        # print("kl: ", kl.shape)
        
        mask_region = (xt == self.mask_id).float()
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1] # 没有被mask的可能被替换，所以权重都是1

        non_pad = (xt != self.pad_id).float()
        
        kl = kl * mask_weight * non_pad
        kl = sum_except_batch(kl)
        # print("kl: ", kl.shape)
        self.log('{}/kl_loss'.format(mode), (kl.sum() / non_pad.sum()).detach(), prog_bar=True, sync_dist=True)

        # decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = -log_categorical(log_x_start, log_x0_recon) # TODO
        decoder_nll = decoder_nll * non_pad
        decoder_nll = sum_except_batch(decoder_nll)
        # print("decoder_nll: ", decoder_nll.shape)
        self.log('{}/decoder_nll'.format(mode), (decoder_nll.sum() / non_pad.sum()).detach(), prog_bar=True, sync_dist=True)

        mask = (t == torch.zeros_like(t)).float() # t=0时，不需要kl loss
        # kl_loss = mask * decoder_nll + (1. - mask) * kl
        kl_loss = (1. - mask) * kl
        
        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        # kl_loss = decoder_nll + (1. - mask) * kl # TODO
        # loss1 = kl_loss / pt
        loss1 = decoder_nll + kl_loss / pt
        vb_loss = loss1.sum() / non_pad.sum()
        self.log('{}/loss'.format(mode), vb_loss.detach(), prog_bar=True, sync_dist=True)

        return log_model_prob, vb_loss
        

    def forward(self, batch, mode):
        """[bs, t, 150]
        """
        self.vqvae.eval()
        gloss_id = batch["gloss_id"]     # [bs, src_len]
        gloss_len = batch["gloss_len"]   # list(src_len)
        points = batch["skel_3d"]        # [bs, max_len, 150]
        skel_len = batch["skel_len"]     # list(skel_len)
        repeat_len = batch["pred_len"]   # list(skel_len)
        gloss = batch["gloss"]           # list(skel_len)
        bs, max_len, v = points.size()
        flat_len = skel_len * 3
        max_flat_len = max(flat_len)
        
        with torch.no_grad():
            points_mask = self._get_mask(skel_len, max_len, points.device)
            _, points_feat, commitment_loss, vq_tokens  = self.vqvae.vqvae_encode(points, points_mask) # [bs, t, 3], [bs, h, t, 3]
            x_0 = einops.rearrange(vq_tokens, "b t n -> b (t n)")
            # print("vq_tokens: ", vq_tokens)
            # print("x_0: ", x_0.shape)
        
        bs = gloss_id.size(0)
        if self.use_discrete_cfg:
            t5_empty_idx =  torch.rand(gloss_id.shape[0]) <= self.empty_text_prob
            gloss_id[t5_empty_idx] = self.text_dict.mask_index

        src_emb = self.gloss_embedding(gloss_id)
        src_emb = self.tem_pos_emb(src_emb)
        src_mask = gloss_id.ne(self.text_dict.pad())
        src_feat = self.encoder(src_emb, src_mask.unsqueeze(1).unsqueeze(2))
        

        flat_len = skel_len * 3
        max_flat_len = max(flat_len)
        x_mask = self._get_mask(flat_len, max_flat_len, vq_tokens.device)
        x_0 = x_0.masked_fill(~x_mask, self.pad_id) # [bs, t]
        log_model_prob, loss = self._train_loss(x_0, x_mask, src_feat, src_mask, mode)

        out = {}
        out['logits'] = torch.exp(log_model_prob)
        out['loss'] = loss
        return out


    def training_step(self, batch, batch_idx):
        out = self.forward(batch, "train")
        if self.lr_schedule is not None:
            self.scheduler.step(self.current_epoch)
        lr = self.optimizer.param_groups[-1]['lr']
        self.log('lr', lr, prog_bar=True, sync_dist=True)
        return out

    
    def p_pred(self, log_x, cond_emb, t, x_mask, src_mask):             # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, cond_emb, t, x_mask, src_mask)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, cond_emb, t)
        else:
            raise ValueError
        return log_model_pred
        
    @torch.no_grad()
    def p_sample(self, log_x, cond_emb, t, x_mask, src_mask):               # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        model_log_prob = self.p_pred(log_x, cond_emb, t, x_mask, src_mask)
        out = self.log_sample_categorical(model_log_prob, x_mask)
        return out

    @torch.no_grad()
    def generate(self, batch, batch_idx):
        gloss_id = batch["gloss_id"]   # [bs, src_len]
        gloss_len = batch["gloss_len"] # list(src_len)
        points = batch["skel_3d"]      # [bs, max_len, 150]
        tgt_len = batch["skel_len"]   # list(skel_len)
        repeat_len = batch["pred_len"]   # list(skel_len)
        gloss = batch["gloss"]   # list(skel_len)
        bs, max_len, v = points.size()
        device = gloss_id.device

        bs = gloss_id.size(0)
        src_emb = self.gloss_embedding(gloss_id)
        src_emb = self.tem_pos_emb(src_emb)
        src_mask = gloss_id.ne(self.text_dict.pad())
        src_feat = self.encoder(src_emb, src_mask.unsqueeze(1).unsqueeze(2))

        unconditional_conditioning_idx = torch.full_like(gloss_id, fill_value=self.text_dict.mask_index)
        unconditional_conditioning_emb = self.gloss_embedding(unconditional_conditioning_idx)
        unconditional_conditioning_emb = self.tem_pos_emb(unconditional_conditioning_emb)
        unconditional_conditioning_emb = self.encoder(unconditional_conditioning_emb, src_mask.unsqueeze(1).unsqueeze(2))

        flat_len = tgt_len * 3
        max_flat_len = max(flat_len)
        x_mask = self._get_mask(flat_len, max_flat_len, device)

        init_x = gloss_id.new(bs, max_flat_len).fill_(self.mask_id)
        init_x = init_x.masked_fill(~x_mask, self.pad_id)
        
        log_z = index_to_log_onehot(init_x, self.num_classes)

        
        start_step = self.num_timesteps
        with torch.no_grad():
            for diffusion_index in range(start_step-1, -1, -1):
                t = torch.full((bs,), diffusion_index, device=device, dtype=torch.long)
                if not self.use_discrete_cfg:
                    model_log_prob = self.p_pred(log_z, src_feat, t, x_mask, src_mask)
                    log_z = self.log_sample_categorical(model_log_prob, x_mask)
                # log_z = self.p_sample(log_z, src_feat, t, x_mask, src_mask)     # log_z is log_onehot
                else:
                    # print("use classifier-free guidance")
                    log_z_in = torch.cat([log_z]*2, dim=0)
                    src_feat_in = torch.cat([src_feat, unconditional_conditioning_emb], dim=0)
                    t_in = torch.cat([t]*2, dim=0)
                    x_mask_in = torch.cat([x_mask]*2, dim=0)
                    src_mask_in = torch.cat([src_mask]*2, dim=0)
                    log_prob_cond, log_prob_uncond = self.p_pred(log_z_in, src_feat_in, t_in, x_mask_in, src_mask_in).chunk(2)
                    model_log_prob = log_prob_uncond + self.unconditional_guidance_scale * (log_prob_cond - log_prob_uncond) # scale = 10
                    log_z = self.log_sample_categorical(model_log_prob, x_mask)
        predictions = log_onehot_to_index(log_z)

        predicts = einops.rearrange(predictions, "b (t n) -> b t n", n=3)

        n_codes, emb_dim = self.vqvae.codebook.embeddings.size()
        embedding = torch.cat([self.vqvae.codebook.embeddings, torch.zeros(2, emb_dim).to(predicts.device)])
        pred_emb = F.embedding(predicts, embedding) # [bs, max_len, 3, emb_dim]
        pred_feat = einops.rearrange(pred_emb, "b t n h -> b h t n")

        dec_mask = self._get_mask(tgt_len, max(tgt_len), pred_feat.device)
        dec_pose, dec_rhand, dec_lhand = self.vqvae.vqvae_decode(pred_feat, dec_mask)

        pred_points = torch.cat([dec_pose, dec_rhand, dec_lhand], dim=-1)
        pred_points = einops.rearrange(pred_points, "(b t) v -> b t v", b=bs) # [b max_len/3 v]

        ori_points = batch["skel_3d"]
        if batch_idx < 2:
            for i in range(bs):
                pred_show_img = []
                ori_show_img = []
                pred_cur_points = pred_points[i, :tgt_len[i].item()].detach().cpu().numpy() # [cur_len, 150]
                ori_cur_points = ori_points[i, :tgt_len[i].item()].detach().cpu().numpy() # [cur_len, 150]
                for j in range(pred_cur_points.shape[0]):
                    frame_joints = pred_cur_points[j]
                    frame = np.ones((512, 512, 3), np.uint8) * 255
                    frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]
                    # Draw the frame given 2D joints
                    im = draw_frame_2D(frame, frame_joints_2d)
                    pred_show_img.append(im)
                for j in range(ori_cur_points.shape[0]):
                    frame_joints = ori_cur_points[j]
                    frame = np.ones((512, 512, 3), np.uint8) * 255
                    frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]
                    # Draw the frame given 2D joints
                    im = draw_frame_2D(frame, frame_joints_2d)
                    ori_show_img.append(im)
                pred_show_img = np.concatenate(pred_show_img, axis=1) # [h, w, c]
                ori_show_img = np.concatenate(ori_show_img, axis=1) # [h, w, c]
                show_img = np.concatenate([pred_show_img, ori_show_img], axis=0)
                save_dir = os.path.join(self.sample_dir, NOW_TIME)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite("{}/epoch={}_batch={}_idx={}.png".format(save_dir, self.current_epoch, batch_idx, i), show_img)
                # show_img = torch.FloatTensor(show_img).permute(2, 0, 1).contiguous().unsqueeze(0) # [1, c, h ,w]
                # show_img = torchvision.utils.make_grid(show_img, )
                # self.logger.experiment.add_image("{}/{}_batch_{}_{}".format("test", "pred", batch_idx, i), show_img, self.global_step)
        return pred_points, tgt_len

    def validation_step(self, batch, batch_idx):
        self.vqvae.eval()
        self.forward(batch, "val")
        gloss_id = batch["gloss_id"]   # [bs, src_len]
        gloss_len = batch["gloss_len"] # list(src_len)
        skel_len = batch["skel_len"]   # list(skel_len)
        ori_points = batch["skel_3d"]

        bs = gloss_id.size(0)

        pred_points, tgt_len = self.generate(batch, batch_idx)
        rec_res1 = self._compute_wer(pred_points, skel_len, gloss_id, gloss_len, "test", self.back_translate_model1)
        ori_res1 = self._compute_wer(ori_points, skel_len, gloss_id, gloss_len, "test", self.back_translate_model1)

        dtw_scores = []
        for i in range(bs):
            dec_point = pred_points[i, :skel_len[i].item(), :].cpu().numpy()
            ori_point = ori_points[i, :skel_len[i].item(), :].cpu().numpy()
            
            euclidean_norm = lambda x, y: np.sum(np.abs(x - y))
            d, cost_matrix, acc_cost_matrix, path = dtw(dec_point, ori_point, dist=euclidean_norm)

            # Normalise the dtw cost by sequence length
            dtw_scores.append(d/acc_cost_matrix.shape[0])
        
        # return rec_res1, ori_res1, dtw_scores, rec_res2, ori_res2
        return rec_res1, ori_res1, dtw_scores


    def validation_epoch_end(self, outputs) -> None:
        rec_err, rec_correct, rec_count = np.zeros([4]), 0, 0
        ori_err, ori_correct, ori_count = np.zeros([4]), 0, 0
        dtw_scores = []
        for rec_out, ori_out, dtw in outputs:
            rec_err += rec_out["wer"]
            rec_correct += rec_out["correct"]
            rec_count += rec_out["count"]
            ori_err += ori_out["wer"]
            ori_correct += ori_out["correct"]
            ori_count += ori_out["count"]
            
            dtw_scores.extend(dtw)

        # self.log('{}/rec_acc'.format("val"), rec_correct / rec_count, prog_bar=True, sync_dist=True)
        self.log('{}/rec_wer'.format("val"), rec_err[0] / rec_count, prog_bar=True, sync_dist=True)
        # self.log('{}/ori_acc'.format("val"), ori_correct / ori_count, prog_bar=True, sync_dist=True)
        self.log('{}/ori_wer'.format("val"), ori_err[0] / ori_count, prog_bar=True, sync_dist=True)
        self.log('{}/rec_dtw'.format("val"), sum(dtw_scores) / len(dtw_scores), prog_bar=True, sync_dist=True)
    
    def points2imgs(self, points, skel_len):
        bs, t, v = points.size()
        video = torch.zeros(bs, t, 3, 128, 128)
        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
        ])
        for b in range(bs):
            cur_points = points[b] # [t, v]
            cur_len = skel_len[b].item()
            cur_imgs = []
            for i in range(cur_len):
                cur_frame = cur_points[i].cpu().numpy() # [150]
                frame = np.ones((256, 256, 3), np.uint8) * 255
                frame_joints_2d = np.reshape(cur_frame, (50, 3))[:, :2]
                # Draw the frame given 2D joints
                im = draw_frame_2D(frame, frame_joints_2d) # [h, w, c]
                im = torch.FloatTensor(im).permute(2,0,1).contiguous()
                im = test_transform(im)
                cur_imgs.append(im.unsqueeze(0))
            cur_imgs = torch.cat(cur_imgs, dim=0)
            video[b, :cur_len, ...] = cur_imgs
        video /= 255.
        return video.cuda()

    def _compute_wer(self, points, skel_len, gloss_id, gloss_len, mode, back_model):
        _, logits = back_model(points, skel_len, gloss_id, gloss_len, mode)
        pred_logits = F.softmax(logits, dim=-1) # [bs, t/4, vocab_size]
        skel_len = torch.div(skel_len, 4, rounding_mode='floor')
        pred_seq, _, _, out_seq_len = back_model.decoder.decode(pred_logits, skel_len)
        
        err_delsubins = np.zeros([4])
        count = 0
        correct = 0
        for i, length in enumerate(gloss_len):
            ref = gloss_id[i][:length].tolist()[:-1]
            hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())][:-1]
            correct += int(ref == hyp)
            err = get_wer_delsubins(ref, hyp)
            err_delsubins += np.array(err)
            count += 1
        res = dict(wer=err_delsubins, correct=correct, count=count)
        return res

    def _get_mask(self, x_len, size, device):
        pos = torch.arange(0, size).unsqueeze(0).repeat(x_len.size(0), 1).to(device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask

    def configure_optimizers(self):
        if self.lr_schedule is None and self.optim_type == "adam" :
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
            return [self.optimizer]
        elif self.lr_schedule is None and self.optim_type == "adamw" :
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
            return [self.optimizer]
        elif self.lr_schedule == "cosineAnnWarm" and self.optim_type == "adamw" :
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=5)
            return [self.optimizer], [self.scheduler]
        else:
            raise ValueError("{} and {} is not selected!".format(self.lr_schedule, self.optim_type) )
    
    def vis_token(self, pred_tokens, mode, name):
        pred_tokens = " ".join([str(x) for x in pred_tokens.cpu().numpy().tolist()])
        self.logger.experiment.add_text("{}/{}_points".format(mode, name), pred_tokens, self.current_epoch)


    def vis(self, pose, rhand, lhand, mode, name, vis_len):
        points = torch.cat([pose, rhand, lhand], dim=-1).detach().cpu().numpy()
        # points: [bs, 150]
        show_img = []
        for j in range(vis_len):
            frame_joints = points[j]
            frame = np.ones((512, 512, 3), np.uint8) * 255
            frame_joints_2d = np.reshape(frame_joints, (50, 3))[:, :2]
            # Draw the frame given 2D joints
            im = draw_frame_2D(frame, frame_joints_2d)
            show_img.append(im)
        show_img = np.concatenate(show_img, axis=1) # [h, w, c]
        show_img = torch.FloatTensor(show_img).permute(2, 0, 1).contiguous().unsqueeze(0) # [1, c, h ,w]
        show_img = torchvision.utils.make_grid(show_img, )
        self.logger.experiment.add_image("{}/{}".format(mode, name), show_img, self.global_step)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--pose_vqvae', type=str, default='kinetics_stride4x4x4', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--backmodel', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--vqvae_hparams_file', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--backmodel_hparams_file', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--embedding_dim', type=int, default=512)
        parser.add_argument('--n_codes', type=int, default=1024)
        parser.add_argument('--n_hiddens', type=int, default=512)
        parser.add_argument('--n_res_layers', type=int, default=2)
        parser.add_argument('--downsample', nargs='+', type=int, default=(4, 4, 4))
        parser.add_argument('--backmodel2', type=str, default='kinetics_stride4x4x4', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--backmodel_hparams_file2', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')

        return parser



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



class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(-2)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb

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


class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0., diffusion_step=100):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(EncoderLayer(dim, heads, mlp_dim, dropout, diffusion_step))

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, diffusion_step):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(heads, dim)
        self.norm2 = nn.LayerNorm(dim)
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


class Decoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0., diffusion_step=100):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(DecoderLayer(dim, heads, mlp_dim, dropout, diffusion_step))

    def forward(self, x, pad_future_mask, enc_out, src_mask, t):
        for layer in self.layers:
            x = layer(x, enc_out, pad_future_mask, src_mask, t)
        return x


# class Transformer(nn.Module):
#     def __init__(self, emb_dim=512, depth=6, block_size=2000):
#         super().__init__()
#         casual_mask = torch.tril(torch.ones(block_size, block_size))
#         self.register_buffer("casual_mask", casual_mask.bool().view(1, 1, block_size, block_size))

#         self.encoder = Encoder(dim=emb_dim, depth=depth, heads=8, mlp_dim=2048, dropout = 0.1)
#         self.decoder = Decoder(dim=emb_dim, depth=depth, heads=8, mlp_dim=2048, dropout = 0.1)


#     def forward(self, src_feat, src_mask, tgt_feat, tgt_mask): 
        
#         enc_out = self.encoder(src_feat, src_mask)
#         bs, t, _ = tgt_feat.size()
#         dec_out = self.decoder(tgt_feat, tgt_mask, enc_out, src_mask)
#         return dec_out


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


if __name__ == "__main__":
    pass