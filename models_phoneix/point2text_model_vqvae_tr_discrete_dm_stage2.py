from email.policy import default
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
    def __init__(self, args, text_dict):
        super().__init__()

        self.text_dict = text_dict

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
        self.gloss_embedding = nn.Embedding(len(text_dict), 512, text_dict.pad())
        self.gloss_embedding.weight.data.normal_(mean=0.0, std=0.02)
        
        self.pad_id = self.vqvae.args.n_codes # 2048
        self.mask_id = self.vqvae.args.n_codes + 1 # 2049
        self.points_vocab_size = self.vqvae.args.n_codes + 2
        self.point_embedding = nn.Embedding(self.points_vocab_size, 512, self.pad_id)
        self.point_embedding.weight.data.normal_(mean=0.0, std=0.02)

        self.tem_pos_emb = PositionalEncoding(0.1, 512, 2000)

        self.transformer = Transformer(emb_dim=512, depth=6, block_size=5000)
        self.out_layer = nn.Linear(512, self.points_vocab_size) 
        self.time_emb = TimeEmbedding(512)
        
        self.num_timesteps = args.total_steps
        self.loss_type = args.loss_type
        self.mask_schedule = args.mask_schedule
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps+1))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps+1))
        self.register_buffer('loss_history', torch.zeros(self.num_timesteps+1))

        assert self.mask_schedule in ['random', 'fixed']

        # back translation
        from .point2text_model_cnn import BackTranslateModel as BackTranslateModel1
        from .point2text_model import BackTranslateModel as BackTranslateModel2
        self.back_translate_model1 = BackTranslateModel1.load_from_checkpoint(args.backmodel, hparams_file=args.backmodel_hparams_file)
        self.back_translate_model2 = BackTranslateModel2.load_from_checkpoint(args.backmodel2, hparams_file=args.backmodel_hparams_file2, strict=False)

        self.random = np.random.RandomState(1234)
        self.save_hyperparameters()


    def compute_ctc(self, dec_feat, skel_len, gloss_id, gloss_len):
        ctc_feat = self.conv(dec_feat)
        ctc_feat = einops.rearrange(ctc_feat, "b h t -> b t h")
        ctc_skel_len = skel_len // 4 
        ctc_logits = self.ctc_out(ctc_feat)  # [bs, t, vocab_size]
        lprobs = ctc_logits.log_softmax(-1) # [b t v] 
        lprobs = einops.rearrange(lprobs, "b t v -> t b v")
        ctc_loss = self.ctcLoss(lprobs.cpu(), gloss_id.cpu(), ctc_skel_len.cpu(), gloss_len.cpu()).to(lprobs.device)
        return ctc_logits, ctc_loss # [b t v], [t b v]
    
    
    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(1, self.num_timesteps+1, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt

        else:
            raise ValueError

    def q_sample(self, x_0, t, tgt_mask):
        # samples q(x_t | x_0)
        # randomly set token to mask with probability t/T
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.rand_like(x_t.float()) < (t.float().unsqueeze(-1) / self.num_timesteps)
        mask[~tgt_mask] = False
        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = self.pad_id
        return x_t, x_0_ignore, mask

    def q_sample_mlm(self, x_0, t):
        # samples q(x_t | x_0)
        # fixed noise schedule, masks exactly int(t/T * latent_size) tokens
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.zeros_like(x_t).to(torch.bool)

        # TODO: offset so each n_masked_tokens is picked with equal probability
        n_masked_tokens = (t.float() / self.num_timesteps) * x_t.size(1)
        n_masked_tokens = torch.round(n_masked_tokens).to(torch.int64)
        n_masked_tokens[n_masked_tokens == 0] = 1
        ones = torch.ones_like(mask[0]).to(torch.bool).to(x_0.device)

        for idx, n_tokens_to_mask in enumerate(n_masked_tokens):
            index = torch.randperm(x_0.size(1))[:n_tokens_to_mask].to(x_0.device)
            mask[idx].scatter_(dim=0, index=index, src=ones)

        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask

    def compute_logits(self, t, gloss_id, tgt_inp, tgt_mask):
        """vq_tokens: [bs, t, 3]
        """
        bs = gloss_id.size(0)
        src_feat = self.gloss_embedding(gloss_id)
        src_feat = self.tem_pos_emb(src_feat)
        src_mask = gloss_id.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)

        tgt_emb_inp = self.point_embedding(tgt_inp) # [b t*n h]
        tgt_emb_inp = self.tem_pos_emb(tgt_emb_inp)
        tgt_emb_inp = tgt_emb_inp + self.time_emb(t).unsqueeze(1)

        out_feat = self.transformer(src_feat, src_mask, tgt_emb_inp, tgt_mask.unsqueeze(1).unsqueeze(2)) # [b t*3 h]

        out_logits = self.out_layer(out_feat) 
        return out_logits


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
            vq_tokens = einops.rearrange(vq_tokens, "b t n -> b (t n)") # [bs, max_len * 3]

            flat_len = skel_len * 3
            max_flat_len = max(flat_len)
            tgt_mask = self._get_mask(flat_len, max_flat_len, vq_tokens.device)
            vq_tokens[~tgt_mask] = self.pad_id


        # choose what time steps to compute loss at
        t, pt = self.sample_time(bs, vq_tokens.device, 'uniform')

        # make x noisy and denoise
        x_0 = vq_tokens
        if self.mask_schedule == 'random':
            x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t, tgt_mask=tgt_mask)
        elif self.mask_schedule == 'fixed':
            x_t, x_0_ignore, mask = self.q_sample_mlm(x_0=x_0, t=t)

        x_0_hat_logits = self.compute_logits(t, gloss_id, x_t, tgt_mask).permute(0, 2, 1).contiguous()

        # Always compute ELBO for comparison purposes
        cross_entropy_loss = F.cross_entropy(x_0_hat_logits, x_0_ignore, ignore_index=self.pad_id, reduction='none').sum(1)
        
        if self.loss_type == 'elbo':
            vb_loss = cross_entropy_loss / t
            vb_loss = vb_loss / pt
            # vb_loss = vb_loss / (math.log(2) * x_0.shape[1:].numel())
            vb_loss = vb_loss / (math.log(2) * skel_len)
            loss = vb_loss
        elif self.loss_type == 'mlm':
            denom = mask.float().sum(1)
            denom[denom == 0] = 1  # prevent divide by 0 errors.
            loss = cross_entropy_loss / denom
        elif self.loss_type == 'reweighted_elbo':
            weight = (1 - (t / self.num_timesteps))
            loss = weight * cross_entropy_loss
            loss = loss / (math.log(2) * skel_len)
        else:
            raise ValueError

        loss = loss.mean()
        self.log('{}_loss'.format(mode), loss.detach(), prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, "train")
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        self.vqvae.eval()
        
        gloss_id = batch["gloss_id"]   # [bs, src_len]
        gloss_len = batch["gloss_len"] # list(src_len)
        skel_len = batch["skel_len"]   # list(skel_len)
        ori_points = batch["skel_3d"]

        bs = gloss_id.size(0)

        pred_points, tgt_len = self.generate(batch, batch_idx, sample_steps=20, use_gold_length=True, temp=1.0,)

        dec_video = self.points2imgs(pred_points, skel_len)
        rec_res1 = self._compute_wer(dec_video, skel_len, gloss_id, gloss_len, "test", self.back_translate_model1)
        rec_res2 = self._compute_wer(pred_points, skel_len, gloss_id, gloss_len, "test", self.back_translate_model2)

        ori_video = self.points2imgs(ori_points, skel_len)
        ori_res1 = self._compute_wer(ori_video, skel_len, gloss_id, gloss_len, "test", self.back_translate_model1)
        ori_res2 = self._compute_wer(ori_points, skel_len, gloss_id, gloss_len, "test", self.back_translate_model2)

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

        # self.log('{}/acc'.format("rec"), rec_correct / rec_count, prog_bar=True)
        self.log('{}_wer'.format("rec"), rec_err[0] / rec_count, prog_bar=True)
        # self.log('{}/acc'.format("ori"), ori_correct / ori_count, prog_bar=True)
        self.log('{}_wer'.format("ori"), ori_err[0] / ori_count, prog_bar=True)
        
        self.log('{}_dtw'.format("test"), sum(dtw_scores) / len(dtw_scores), prog_bar=True)

        # self.log('{}/acc2'.format("rec2"), rec_correct2 / rec_count2, prog_bar=True)
        self.log('{}_wer2'.format("rec2"), rec_err2[0] / rec_count2, prog_bar=True)
        # self.log('{}/acc2'.format("ori2"), ori_correct2 / ori_count2, prog_bar=True)
        self.log('{}_wer2'.format("ori2"), ori_err2[0] / ori_count2, prog_bar=True)
    
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