from turtle import forward
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
import torch.distributions as dists
from util.ddpm_utils import make_beta_schedule, extract_into_tensor, default, noise_like
from functools import partial
from tqdm import tqdm


class Swish(nn.Module):
    """
    ### Swish actiavation function
    $$x \cdot \sigma(x)$$
    """
    def forward(self, x):
        # F.gumbel_softmax
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb

class Point2textModelStage2(pl.LightningModule):
    def __init__(self, args, text_dict, 
        given_betas=None, beta_schedule="linear", timesteps=100, linear_start=1e-4, 
        linear_end=2e-2, cosine_s=8e-3, v_posterior=0., original_elbo_weight=0., l_simple_weight=1.,
        parameterization="eps", loss_type="l1") :
        super().__init__()

        self.text_dict = text_dict
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight
        self.parameterization = parameterization
        self.loss_type = loss_type

        # vqvae
        from .point2text_model_vqvae_tr_nat_stage1_seperate2 import Point2textModel
        if not os.path.exists(args.pose_vqvae):
            raise ValueError("{} is not existed!".format(args.pose_vqvae))
        else:
            print("load vqvae model from {}".format(args.pose_vqvae))
            self.vqvae =  Point2textModel.load_from_checkpoint(args.pose_vqvae, hparams_file=args.vqvae_hparams_file, strict=False)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        # encoder-decoder
        self.gloss_embedding = nn.Embedding(len(text_dict), 256, text_dict.pad())
        self.gloss_embedding.weight.data.normal_(mean=0.0, std=0.02)
        
        self.tem_pos_emb = nn.Parameter(torch.zeros(2000, 256))
        self.tem_pos_emb.data.normal_(0, 0.02)
        self.transformer = Transformer(emb_dim=256, depth=6, block_size=5000)
        self.time_emb = TimeEmbedding(256)
        
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        # back translation
        # from .point2text_model_cnn import BackTranslateModel as BackTranslateModel1
        # from .point2text_model import BackTranslateModel as BackTranslateModel2
        # self.back_translate_model1 = BackTranslateModel1.load_from_checkpoint(args.backmodel, hparams_file=args.backmodel_hparams_file)
        # self.back_translate_model2 = BackTranslateModel2.load_from_checkpoint(args.backmodel2, hparams_file=args.backmodel_hparams_file2, strict=False)

        self.random = np.random.RandomState(1234)
        self.save_hyperparameters()

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas', to_torch(alphas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion p_{theta}(x_{t-1} | x_t) and others
        self.register_buffer('eps_coef', to_torch(betas/ np.sqrt(1. - alphas_cumprod)))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def compute_ctc(self, dec_feat, skel_len, gloss_id, gloss_len):
        ctc_feat = self.conv(dec_feat)
        ctc_feat = einops.rearrange(ctc_feat, "b h t -> b t h")
        ctc_skel_len = skel_len // 4 
        ctc_logits = self.ctc_out(ctc_feat)  # [bs, t, vocab_size]
        lprobs = ctc_logits.log_softmax(-1) # [b t v] 
        lprobs = einops.rearrange(lprobs, "b t v -> t b v")
        ctc_loss = self.ctcLoss(lprobs.cpu(), gloss_id.cpu(), ctc_skel_len.cpu(), gloss_len.cpu()).to(lprobs.device)
        return ctc_logits, ctc_loss # [b t v], [t b v]
    
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def model_out(self, t, feat_inp, feat_mask, gloss_id, gloss_len):
        """x_noisy: [bs, h, t, 3]
        """
        bs = gloss_id.size(0)
        src_feat = self.gloss_embedding(gloss_id)
        src_feat = src_feat + self.tem_pos_emb[:src_feat.size(1)]
        src_mask = gloss_id.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)
        feat_inp = feat_inp + self.tem_pos_emb[:feat_inp.size(1)] # TODO
        feat_inp = feat_inp + self.time_emb(t).unsqueeze(1)
        out_feat = self.transformer(src_feat, src_mask, feat_inp, feat_mask.unsqueeze(1).unsqueeze(2)) # [b t*3 h]
        return out_feat

    def p_losses(self, t, x_start, skel_len, gloss_id, gloss_len, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        feat_inp = einops.rearrange(x_noisy, "b h t n -> b (t n) h")
        flat_len = skel_len * 3
        max_flat_len = max(flat_len)
        feat_mask = self._get_mask(flat_len, max_flat_len, x_noisy.device)
        model_out = self.model_out(t, feat_inp, feat_mask, gloss_id, gloss_len)
        model_out = einops.rearrange(model_out, "b (t n) h -> b h t n", n=3)
        
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False)

        loss = einops.rearrange(loss, "b h t n -> b h (t n)")
        loss = loss * feat_mask.unsqueeze(1) # [b h (t*n)]

        loss = loss.sum(dim=[1, 2]) / feat_mask.sum(-1)

        log_prefix = 'train' if self.training else 'val'
        loss_simple = loss.mean() * self.l_simple_weight
        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss = loss_simple + self.original_elbo_weight * loss_vlb

        self.log('{}/loss'.format(log_prefix), loss.detach(), prog_bar=True)
        self.log('{}/loss_vlb'.format(log_prefix), loss_vlb.detach(), prog_bar=True)
        self.log('{}/loss_simple'.format(log_prefix), loss_simple.detach(), prog_bar=True)
        return loss

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        return loss


    def forward(self, batch, mode):
        """[bs, t, 150]
        """
        self.vqvae.eval()
        gloss_id = batch["gloss_id"]   # [bs, src_len]
        gloss_len = batch["gloss_len"] # list(src_len)
        points = batch["skel_3d"]      # [bs, max_len, 150]
        skel_len = batch["skel_len"]   # list(skel_len)
        bs, max_len, v = points.size()

        with torch.no_grad():
            points_mask = self._get_mask(skel_len, max_len, points.device)
            vq_tokens, points_feat, commitment_loss = self.vqvae.vqvae_encode(points, points_mask) # [bs, t, 3], [bs, h, t, 3]
            x = points_feat
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(t, x, skel_len, gloss_id, gloss_len)
        

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, "train")
        return loss
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(self, t, x, x_mask, gloss_id, gloss_len):
        model_out = self.model_out(t, x, x_mask, gloss_id, gloss_len)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, x_mask, gloss_id, gloss_len):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(t, x, x_mask, gloss_id, gloss_len)
        noise = noise_like(x.shape, device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample2(self, xt, t, eps_theta):
        alpha = extract_into_tensor(self.alphas, t, xt.shape)
        eps_coef = extract_into_tensor(self.eps_coef, t, xt.shape)
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = extract_into_tensor(self.betas, t, xt.shape)
        eps = torch.randn(xt.shape, device=xt.device, dtype=torch.float32)
        return mean + (var ** .5) * eps

        
    @torch.no_grad()
    def p_sample_loop(self, batch, verbose=False):
        
        device = self.betas.device

        self.vqvae.eval()
        gloss_id = batch["gloss_id"]   # [bs, src_len]
        gloss_len = batch["gloss_len"] # list(src_len)
        points = batch["skel_3d"]      # [bs, max_len, 150]
        skel_len = batch["skel_len"]   # list(skel_len)

        timesteps = self.num_timesteps

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        src_feat = self.gloss_embedding(gloss_id)
        src_feat = src_feat + self.tem_pos_emb[:src_feat.size(1)]
        src_mask = gloss_id.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)

        b = gloss_id.size(0)
        xt = torch.randn((b, 256, max(skel_len), 3), device=device)
        feat_inp = einops.rearrange(xt, "b h t n -> b (t n) h")
        flat_len = skel_len * 3
        max_flat_len = max(flat_len)
        feat_mask = self._get_mask(flat_len, max_flat_len, feat_inp.device)

        for i in iterator:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            feat_inp = feat_inp + self.tem_pos_emb[:feat_inp.size(1)] # TODO
            feat_inp = feat_inp + self.time_emb(t).unsqueeze(1)
            feat_inp = self.p_sample(feat_inp, t, feat_mask, gloss_id, gloss_len)
        return feat_inp, feat_mask


    def validation_step(self, batch, batch_idx):
        gloss_id = batch["gloss_id"]   # [bs, src_len]
        gloss_len = batch["gloss_len"] # list(src_len)
        skel_len = batch["skel_len"]   # list(skel_len)
        ori_points = batch["skel_3d"]

        bs = gloss_id.size(0)


        feat_out, _ = self.p_sample_loop(batch)
        max_len = max(skel_len)
        feat_mask = self._get_mask(skel_len, max_len, feat_out.device)
        feat_out = einops.rearrange(feat_out, "b (t n) h -> b h t n", n=3)

        pred_pose, pred_rhand, pred_lhand = self.vqvae.vqvae_decode(feat_out, feat_mask)
        pred_points = torch.cat([pred_pose, pred_rhand, pred_lhand], dim=-1) # [bs*max_len, 150]

        pred_points = einops.rearrange(pred_points, "(b t) h -> b t h", b=bs)
        # print("pred_points: ", pred_points.shape)

        dec_video = self.points2imgs(pred_points, skel_len)
        rec_res1 = self._compute_wer(dec_video, skel_len, gloss_id, gloss_len, "test", self.vqvae.back_translate_model1)
        rec_res2 = self._compute_wer(pred_points, skel_len, gloss_id, gloss_len, "test", self.vqvae.back_translate_model2)

        ori_video = self.points2imgs(ori_points, skel_len)
        ori_res1 = self._compute_wer(ori_video, skel_len, gloss_id, gloss_len, "test", self.vqvae.back_translate_model1)
        ori_res2 = self._compute_wer(ori_points, skel_len, gloss_id, gloss_len, "test", self.vqvae.back_translate_model2)

        dtw_scores = []
        for i in range(bs):
            dec_point = pred_points[i, :skel_len[i].item(), :].cpu().numpy()
            ori_point = ori_points[i, :skel_len[i].item(), :].cpu().numpy()
            
            euclidean_norm = lambda x, y: np.sum(np.abs(x - y))
            d, cost_matrix, acc_cost_matrix, path = dtw(dec_point, ori_point, dist=euclidean_norm)

            # Normalise the dtw cost by sequence length
            dtw_scores.append(d/acc_cost_matrix.shape[0])
        
        return rec_res1, ori_res1, dtw_scores, rec_res2, ori_res2


    def validation_epoch_end(self, outputs) -> None:
        rec_err, rec_correct, rec_count = np.zeros([4]), 0, 0
        ori_err, ori_correct, ori_count = np.zeros([4]), 0, 0
        rec_err2, rec_correct2, rec_count2 = np.zeros([4]), 0, 0
        ori_err2, ori_correct2, ori_count2 = np.zeros([4]), 0, 0
        dtw_scores = []
        for rec_out, ori_out, dtw, rec_out2, ori_out2, in outputs:
            rec_err += rec_out["wer"]
            rec_correct += rec_out["correct"]
            rec_count += rec_out["count"]
            ori_err += ori_out["wer"]
            ori_correct += ori_out["correct"]
            ori_count += ori_out["count"]
            
            dtw_scores.extend(dtw)
            
            rec_err2 += rec_out2["wer"]
            rec_correct2 += rec_out2["correct"]
            rec_count2 += rec_out2["count"]
            ori_err2 += ori_out2["wer"]
            ori_correct2 += ori_out2["correct"]
            ori_count2 += ori_out2["count"]

        self.log('{}/rec_wer'.format("val"), rec_err[0] / rec_count, prog_bar=True)
        self.log('{}/ori_wer'.format("val"), ori_err[0] / ori_count, prog_bar=True)
        self.log('{}/dtw'.format("val"), sum(dtw_scores) / len(dtw_scores), prog_bar=True)
        self.log('{}/rec_wer2'.format("val"), rec_err2[0] / rec_count2, prog_bar=True)
        self.log('{}/ori_wer2'.format("val"), ori_err2[0] / ori_count2, prog_bar=True)
    
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
        skel_len = skel_len // 4
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

    @torch.no_grad()
    def generate(self, batch, batch_idx, sample_steps=None, use_gold_length=True, temp=1.0,):
        gloss_id = batch["gloss_id"]
        gloss_len = batch["gloss_len"]
        skel_len = batch["skel_len"]

        bs, max_src_len = gloss_id.size()
        src_feat = self.gloss_embedding(gloss_id)
        src_feat = self.tem_pos_emb(src_feat)
        src_mask = gloss_id.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)

        # use ground truth length
        if use_gold_length:
            tgt_len = skel_len
            flat_len = tgt_len * 3
            max_len = max(flat_len)
        else:
            tgt_len = gloss_len * 8
            flat_len = tgt_len * 3
            max_len = max(flat_len)
        
        # initilize target input
        device = src_feat.device
        x_t = gloss_id.new(bs, max_len).fill_(self.mask_id)  # [bs, max_len*3]
        tgt_mask = self._get_mask(flat_len, max_len, gloss_id.device) # [bs, max_len*3]
        x_t = tgt_mask.long() * x_t + (1 - tgt_mask.long()) * self.pad_id
        

        unmasked = torch.zeros_like(x_t, device=device).bool()
        sample_steps = list(range(1, sample_steps+1))
        
        for t in reversed(sample_steps):
            t = torch.full((bs,), t, device=device, dtype=torch.long)
            # where to unmask
            changes = torch.rand(x_t.shape, device=device) < 1/t.float().unsqueeze(-1)
            changes = changes & tgt_mask
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)
            x_0_logits = self.compute_logits(t, gloss_id, x_t, tgt_mask.clone())
            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_t[changes] = x_0_hat[changes]

        predicts = einops.rearrange(x_t, "b (t n) -> b t n", n=3)

        n_codes, emb_dim = self.vqvae.codebook.embeddings.size()
        embedding = torch.cat([self.vqvae.codebook.embeddings, torch.zeros(2, emb_dim).to(predicts.device)])
        pred_emb = F.embedding(predicts, embedding) # [bs, max_len, 3, emb_dim]
        pred_feat = einops.rearrange(pred_emb, "b t n h -> b h t n")

        dec_mask = self._get_mask(tgt_len, max(tgt_len), gloss_id.device)
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
                cv2.imwrite("/Dataset/everybody_sign_now_experiments/predictions2/epoch={}_batch={}_idx={}.png".format(self.current_epoch, batch_idx, i), show_img)
                # show_img = torch.FloatTensor(show_img).permute(2, 0, 1).contiguous().unsqueeze(0) # [1, c, h ,w]
                # show_img = torchvision.utils.make_grid(show_img, )
                # self.logger.experiment.add_image("{}/{}_batch_{}_{}".format("test", "pred", batch_idx, i), show_img, self.global_step)
        return pred_points, tgt_len

    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)

    @torch.no_grad()
    def generate_step_with_probs(self, tgt_inp, tgt_len, src_feat, src_mask):
        """tgt_inp: [bs, t, 3]
           tgt_mask: [bs, t]
        """
        b = tgt_inp.size(0)

        tgt_emb_inp = self.point_embedding(tgt_inp) # [b t*n h]
        tgt_emb_inp = self.tem_pos_emb(tgt_emb_inp)

        flat_len = tgt_len * 3
        max_flat_len = max(flat_len)
        tgt_mask = self._get_mask(flat_len, max_flat_len, tgt_inp.device).unsqueeze_(1).unsqueeze_(2)
        out_feat = self.transformer(src_feat, src_mask, tgt_emb_inp, tgt_mask) # [b t*3 h]

        out_logits = self.out_layer(out_feat) 
        out_logits = einops.rearrange(out_logits, "b t v -> (b t) v")


        out_logits = self.out_layer(out_feat) 
        out_logits = einops.rearrange(out_logits, "(b t) n v -> b t n v", b=b) # [b*t n h]

        out_logits[:, :, :, -2:] = float("-inf")
        probs = F.softmax(out_logits, dim=-1)
        max_probs, idx = probs.max(dim=-1)
        return idx, max_probs # [bs, max_len, 3]

    def _get_mask(self, x_len, size, device):
        pos = torch.arange(0, size).unsqueeze(0).repeat(x_len.size(0), 1).to(device)
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 3, gamma=0.96, last_epoch=-1)
        return [self.optimizer], [scheduler]
    
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
        parser.add_argument('--pose_vqvae', type=str, default='', help='')
        parser.add_argument('--backmodel', type=str, default='', help='')
        parser.add_argument('--vqvae_hparams_file', type=str, default='', help='')
        parser.add_argument('--backmodel_hparams_file', type=str, default='', help='')
        parser.add_argument('--embedding_dim', type=int, default=512)
        parser.add_argument('--n_codes', type=int, default=1024)
        parser.add_argument('--n_hiddens', type=int, default=512)
        parser.add_argument('--n_res_layers', type=int, default=2)
        parser.add_argument('--downsample', nargs='+', type=int, default=(4, 4, 4))
        parser.add_argument('--backmodel2', type=str, default='', help='')
        parser.add_argument('--backmodel_hparams_file2', type=str, default='', help='')
        
        # ldm
        parser.add_argument('--total_steps', type=int, default=100, help='')
        parser.add_argument('--loss_type', type=str, default="reweighted_elbo", help='reweighted_elbo or mlm or elbo')
        parser.add_argument('--mask_schedule', type=str, default="fixed", help='random or fixed')

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
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(EncoderLayer(dim, heads, mlp_dim, dropout))

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

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

class DecoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        
        self.self_attn = Attention(heads, dim)
        self.cross_attn = Attention(heads, dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout = dropout)

        self.norm1 = BertLayerNorm(dim)
        self.norm2  = BertLayerNorm(dim)
        self.norm3  = BertLayerNorm(dim)

        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x, enc_out, tgt_mask, src_mask):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x
        return x


class Decoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(DecoderLayer(dim, heads, mlp_dim, dropout))

    def forward(self, x, pad_future_mask, enc_out, src_mask):
        for layer in self.layers:
            x = layer(x, enc_out, pad_future_mask, src_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, emb_dim=512, depth=6, block_size=2000):
        super().__init__()
        casual_mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("casual_mask", casual_mask.bool().view(1, 1, block_size, block_size))
        self.encoder = Encoder(dim=emb_dim, depth=depth, heads=8, mlp_dim=2048, dropout = 0.1)
        self.decoder = Decoder(dim=emb_dim, depth=depth, heads=8, mlp_dim=2048, dropout = 0.1)


    def forward(self, src_feat, src_mask, tgt_feat, tgt_mask): 
        
        enc_out = self.encoder(src_feat, src_mask)
        bs, t, _ = tgt_feat.size()
        # casual_mask = self.casual_mask[:, :, :t, :t]
        # if tgt_mask is not None:
        #     pad_future_mask = casual_mask & tgt_mask
        # else:
        #     pad_future_mask = casual_mask
        dec_out = self.decoder(tgt_feat, tgt_mask, enc_out, src_mask)
        return dec_out


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