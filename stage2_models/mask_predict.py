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
from data.vocabulary import Dictionary
from util.train_utils import instantiate_from_config
import time
from util.phoneix_cleanup import clean_phoenix_2014
NOW_TIME = "-".join(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())[5:-3].split(":")).replace(" ", "-")

class Point2textModelStage2(pl.LightningModule):
    def __init__(self,
            vocab_file,
            stage1_model_config,
            backtrans_model_config,
            embedding_dim,
            depth,
            heads,
            dropout,
            resume_ckpt,
            learning_rate,
            sample_dir,
            ):
        super().__init__()
        
        text_dict = Dictionary()
        text_dict = text_dict.load(vocab_file)
        self.text_dict = text_dict
        self.resume_ckpt = resume_ckpt
        self.learning_rate = learning_rate
        self.sample_dir = sample_dir

        # vqvae
        pose_vqvae = instantiate_from_config(stage1_model_config, init_model=False)
        from stage1_models.pose_vqvae import PoseVQVAE
        if not os.path.exists(stage1_model_config.ckpt_path):
            raise ValueError("{} is not existed!".format(stage1_model_config.ckpt_path))
        else:
            print("load vqvae model from {}".format(stage1_model_config.ckpt_path))
            self.vqvae =  pose_vqvae.load_from_checkpoint(stage1_model_config.ckpt_path)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        back_translate_model1 = instantiate_from_config(backtrans_model_config, init_model=False)
        if not os.path.exists(backtrans_model_config.ckpt_path):
            raise ValueError("{} is not existed!".format(backtrans_model_config.ckpt_path))
        else:
            print("load vqvae model from {}".format(backtrans_model_config.ckpt_path))
            self.back_translate_model1 =  back_translate_model1.load_from_checkpoint(backtrans_model_config.ckpt_path)
        for p in self.back_translate_model1.parameters():
            p.requires_grad = False
        self.back_translate_model1.eval()

        # encoder-decoder
        self.gloss_embedding = nn.Embedding(len(text_dict), embedding_dim, text_dict.pad())
        self.gloss_embedding.weight.data.normal_(mean=0.0, std=0.02)
        
        self.pad_id = self.vqvae.args.n_codes
        self.mask_id = self.vqvae.args.n_codes + 1
        self.points_vocab_size = self.vqvae.args.n_codes + 2
        self.point_embedding = nn.Embedding(self.points_vocab_size, embedding_dim, self.pad_id)
        self.point_embedding.weight.data.normal_(mean=0.0, std=0.02)

        # self.tem_pos_emb = nn.Parameter(torch.zeros(2000, 512))
        # self.tem_pos_emb.data.normal_(0, 0.02)
        self.tem_pos_emb = PositionalEncoding(0.1, embedding_dim, 2000)

        # self.transformer = Transformer(emb_dim=512, depth=6, block_size=5000)
        self.encoder = Encoder(dim=embedding_dim, depth=depth, heads=heads, mlp_dim=embedding_dim*4, dropout = dropout)
        self.decoder = Decoder(dim=embedding_dim, depth=depth, heads=heads, mlp_dim=embedding_dim*4, dropout = dropout)
        self.out_layer = nn.Linear(embedding_dim, self.points_vocab_size) 

        self.random = np.random.RandomState(1234)
        self.save_hyperparameters()
        self.resume()

    def resume(self, ):
        if os.path.exists(self.resume_ckpt):
            print("=== Load from {}!".format(self.resume_ckpt))
            self.load_from_checkpoint(self.resume_ckpt, strict=True)
        else:
            print("=== {} is not existed!".format(self.resume_ckpt))


    def compute_ctc(self, dec_feat, skel_len, gloss_id, gloss_len):
        ctc_feat = self.conv(dec_feat)
        ctc_feat = einops.rearrange(ctc_feat, "b h t -> b t h")
        # ctc_skel_len = skel_len // 4 
        ctc_skel_len = torch.div(skel_len, 4, rounding_mode="floor")
        ctc_logits = self.ctc_out(ctc_feat)  # [bs, t, vocab_size]
        lprobs = ctc_logits.log_softmax(-1) # [b t v] 
        lprobs = einops.rearrange(lprobs, "b t v -> t b v")
        ctc_loss = self.ctcLoss(lprobs.cpu(), gloss_id.cpu(), ctc_skel_len.cpu(), gloss_len.cpu()).to(lprobs.device)
        return ctc_logits, ctc_loss # [b t v], [t b v]


    def compute_seq2seq_ce(self, gloss_id, gloss_len, vq_tokens, skel_len, repeat_len):
        """vq_tokens: [bs, t, 3]
        """
        
        bs = gloss_id.size(0)
        src_emb = self.gloss_embedding(gloss_id)
        src_emb = self.tem_pos_emb(src_emb)
        src_mask = gloss_id.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)

        src_feat = self.encoder(src_emb, src_mask)

        new_src_feat = torch.zeros(bs, max(skel_len), src_emb.size(-1)).to(gloss_id.device)
        for i in range(bs):
            start = 0
            for j in range(gloss_len[i].item()):
                end = start + repeat_len[i][j]
                new_src_feat[i, start:end] = src_feat[i, j]
                start = end
        new_src_feat = einops.repeat(new_src_feat, "b t h -> b (t n) h", n=3)

        flat_len = skel_len * 3
        new_vq_tokens = einops.rearrange(vq_tokens, "b t n -> b (t n)") # [bs, max_len * 3]
        tgt_inp = []
        tgt_out = []
        for i in range(bs):
            cur_len = flat_len[i].item()
            new_vq_tokens[i, cur_len:] = self.pad_id
            cur_point = new_vq_tokens[i]

            sample_size = self.random.randint(cur_len*0.5, cur_len)
            ind = self.random.choice(cur_len, size=sample_size, replace=False)

            cur_inp = cur_point.clone()
            cur_inp[ind] = self.mask_id
            tgt_inp.append(cur_inp.unsqueeze_(0))

            cur_out = torch.ones_like(cur_point).to(cur_point.device) * self.pad_id
            cur_out[ind] = cur_point[ind]
            tgt_out.append(cur_out.unsqueeze_(0))

        tgt_inp = torch.cat(tgt_inp, dim=0)  # [b t*n]
        tgt_out = torch.cat(tgt_out, dim=0)
        # print("new_vq_tokens: ", new_vq_tokens[:2, :20], tgt_inp.shape)
        # print("tgt_inp: ", tgt_inp[:2, :10], tgt_inp.shape)
        # print("tgt_out: ", tgt_out[:2, :10], tgt_out.shape)

        tgt_emb_inp = self.point_embedding(tgt_inp) # [b t*n h]
        tgt_emb_inp = self.tem_pos_emb(tgt_emb_inp)

        max_flat_len = max(flat_len)
        tgt_mask = self._get_mask(flat_len, max_flat_len, vq_tokens.device).unsqueeze(1).unsqueeze(2)
        out_feat = self.decoder(tgt_emb_inp, tgt_mask, new_src_feat, tgt_mask) # [b t*3 h]                           

        out_logits = self.out_layer(out_feat) 
        out_logits = einops.rearrange(out_logits, "b t v -> (b t) v")
        tgt_out = tgt_out.view(-1)
        
        ce_loss = F.cross_entropy(out_logits, tgt_out, ignore_index=self.pad_id, reduction="none")
        non_pad = tgt_out.ne(self.pad_id)
        ce_loss = ce_loss.sum() / non_pad.sum()
        return ce_loss

    def forward(self, batch, mode):
        """[bs, t, 150]
        """
        self.vqvae.eval()
        gloss_id = batch["gloss_id"]   # [bs, src_len]
        gloss_len = batch["gloss_len"] # list(src_len)
        points = batch["skel_3d"]      # [bs, max_len, 150]
        skel_len = batch["skel_len"]   # list(skel_len)
        repeat_len = batch["pred_len"]   # list(skel_len)
        # repeat_len = batch["skel_len"]
        bs, max_len, v = points.size()

        with torch.no_grad():
            points_mask = self._get_mask(skel_len, max_len, points.device)
            vq_tokens, points_feat, commitment_loss = self.vqvae.vqvae_encode(points, points_mask) # [bs, t, 3], [bs, h, t, 3]
        
        ce_loss = self.compute_seq2seq_ce(gloss_id, gloss_len, vq_tokens, skel_len, repeat_len)
        # self.log('{}_ce_loss'.format(mode), ce_loss.detach(), prog_bar=True)

        loss = ce_loss
        self.log('{}/loss'.format(mode), loss.detach(), prog_bar=True, sync_dist=True)
        return loss


    def training_step(self, batch):
        loss = self.forward(batch, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.vqvae.eval()
        self.forward(batch, "val")
        
        gloss_id = batch["gloss_id"]   # [bs, src_len]
        gloss_len = batch["gloss_len"] # list(src_len)
        skel_len = batch["skel_len"]   # list(skel_len)
        ori_points = batch["skel_3d"]

        bs = gloss_id.size(0)

        # pred_points_5, tgt_len = self.generate(batch, batch_idx, iterations=5)
        pred_points, tgt_len = self.generate(batch, batch_idx, iterations=1)
        # dec_video = self.points2imgs(pred_points, skel_len)
        # rec_res1 = self._compute_wer(dec_video, skel_len, gloss_id, gloss_len, "test", self.vqvae.back_translate_model1)
        rec_res1 = self._compute_wer(pred_points, skel_len, gloss_id, gloss_len, "test", self.back_translate_model1)

        # ori_video = self.points2imgs(ori_points, skel_len)
        # ori_res1 = self._compute_wer(ori_video, skel_len, gloss_id, gloss_len, "test", self.vqvae.back_translate_model1)
        ori_res1 = self._compute_wer(ori_points, skel_len, gloss_id, gloss_len, "test", self.back_translate_model1)

        dtw_scores = []
        for i in range(bs):
            dec_point = pred_points[i, :skel_len[i].item(), :].cpu().numpy()
            ori_point = ori_points[i, :skel_len[i].item(), :].cpu().numpy()
            
            euclidean_norm = lambda x, y: np.sum(np.abs(x - y))
            d, cost_matrix, acc_cost_matrix, path = dtw(dec_point, ori_point, dist=euclidean_norm)

            # Normalise the dtw cost by sequence length
            dtw_scores.append(d/acc_cost_matrix.shape[0])
        
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

        self.log('{}/rec_wer'.format("val"), rec_err[0] / rec_count, prog_bar=True, sync_dist=True)
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
        # skel_len = skel_len // 4
        skel_len = torch.div(skel_len, 4, rounding_mode="floor")
        pred_seq, _, _, out_seq_len = back_model.decoder.decode(pred_logits, skel_len)
        
        err_delsubins = np.zeros([4])
        count = 0
        correct = 0
        for i, length in enumerate(gloss_len):
            ref = gloss_id[i][:length].tolist()[:-1] # [:-1], append eos=False
            hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())][:-1]
            # ref_sent = clean_phoenix_2014(self.text_dict.deocde_list(ref))
            # hyp_sent = clean_phoenix_2014(self.text_dict.deocde_list(hyp))
            correct += int(ref == hyp)
            err = get_wer_delsubins(ref, hyp)
            err_delsubins += np.array(err)
            count += 1
        res = dict(wer=err_delsubins, correct=correct, count=count)
        return res

    @torch.no_grad()
    def generate(self, batch, batch_idx, iterations=10, use_gold_length=True):
        gloss_id = batch["gloss_id"]
        gloss_len = batch["gloss_len"]
        skel_len = batch["skel_len"]
        repeat_len = batch["pred_len"]   # list(skel_len)

        bs, max_src_len = gloss_id.size()
        src_emb = self.gloss_embedding(gloss_id)
        src_emb = self.tem_pos_emb(src_emb)
        src_mask = gloss_id.ne(self.text_dict.pad()).unsqueeze_(1).unsqueeze_(2)

        src_feat = self.encoder(src_emb, src_mask)

        new_src_feat = torch.zeros(bs, max(skel_len), src_emb.size(-1)).to(gloss_id.device)
        for i in range(bs):
            start = 0
            for j in range(gloss_len[i].item()):
                end = start + repeat_len[i][j]
                new_src_feat[i, start:end] = src_feat[i, j]
                start = end
        new_src_feat = einops.repeat(new_src_feat, "b t h -> b (t n) h", n=3)

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
        tgt_inp = gloss_id.new(bs, max_len).fill_(self.mask_id)  # [bs, max_len*3]
        tgt_mask = self._get_mask(flat_len, max_len, gloss_id.device) # [bs, max_len*3]
        tgt_inp = tgt_mask.long() * tgt_inp + (1 - tgt_mask.long()) * self.pad_id
        pad_mask = tgt_inp.eq(self.pad_id)

        # transformer
        tgt_emb_inp = self.point_embedding(tgt_inp) # [b t*n h]
        tgt_emb_inp = self.tem_pos_emb(tgt_emb_inp)

        # out_feat = self.transformer(src_feat, src_mask, tgt_emb_inp, tgt_mask.unsqueeze(1).unsqueeze(2)) # [b t*3 h]
        # print("tgt_emb_inp, tgt_mask, new_src_feat, tgt_mask: ", tgt_emb_inp.shape, tgt_mask.shape, new_src_feat.shape, tgt_mask.shape)
        tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(2)
        out_feat = self.decoder(tgt_emb_inp, tgt_mask, new_src_feat, tgt_mask) # [b t*3 h]                           

        out_logits = self.out_layer(out_feat)  # [b t*3 vocab]
        probs = F.softmax(out_logits, dim=-1)
        tgt_probs, tgt_inp = probs.max(dim=-1) # [b t*3]


        assign_single_value_byte(tgt_inp, pad_mask, self.pad_id)
        assign_single_value_byte(tgt_probs, pad_mask, 1.0)
        predicts = einops.rearrange(tgt_inp, "b (t n) -> b t n", n=3)

        n_codes, emb_dim = self.vqvae.codebook.embeddings.size()
        embedding = torch.cat([self.vqvae.codebook.embeddings, torch.zeros(2, emb_dim).to(predicts.device)])
        pred_emb = F.embedding(predicts, embedding) # [bs, max_len, 3, emb_dim]
        pred_feat = einops.rearrange(pred_emb, "b t n h -> b h t n")

        dec_mask = self._get_mask(tgt_len, max(tgt_len), gloss_id.device)
        dec_pose, dec_rhand, dec_lhand = self.vqvae.vqvae_decode(pred_feat, dec_mask)

        pred_points = torch.cat([dec_pose, dec_rhand, dec_lhand], dim=-1)
        pred_points = einops.rearrange(pred_points, "(b t) v -> b t v", b=bs) # [b max_len/3 v]

        ori_points = batch["skel_3d"]

        pred_img = self.points2imgs(pred_points, tgt_len) * 255.
        ori_img = self.points2imgs(ori_points, tgt_len) * 255.

        for i in range(bs):
            name = self.text_dict.deocde_list(gloss_id[i, :gloss_len[i]])
            pred_show = einops.rearrange(pred_img[i], "t c h w -> h (t w) c").cpu().numpy()
            ori_show = einops.rearrange(ori_img[i], "t c h w -> h (t w) c").cpu().numpy()
            show_img = np.concatenate([ori_show, pred_show], axis=0)
            save_dir = os.path.join(self.sample_dir, NOW_TIME)
            if os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite("{}/epoch={}_batch={}_idx={}.png".format(save_dir, self.current_epoch, batch_idx, i), show_img)
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 3, gamma=0.96, last_epoch=-1)
        # return [self.optimizer], [scheduler]
        return [self.optimizer]
    
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
        # parser.add_argument('--pose_vqvae', type=str, default='kinetics_stride4x4x4', help='path to vqvae ckpt, or model name to download pretrained')
        # parser.add_argument('--backmodel', type=str, default='kinetics_stride4x4x4', help='path to vqvae ckpt, or model name to download pretrained')
        # parser.add_argument('--vqvae_hparams_file', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')
        # parser.add_argument('--backmodel_hparams_file', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--embedding_dim', type=int, default=512)
        parser.add_argument('--n_codes', type=int, default=1024)
        parser.add_argument('--n_hiddens', type=int, default=512)
        parser.add_argument('--n_res_layers', type=int, default=2)
        parser.add_argument('--downsample', nargs='+', type=int, default=(4, 4, 4))
        # parser.add_argument('--backmodel2', type=str, default='kinetics_stride4x4x4', help='path to vqvae ckpt, or model name to download pretrained')
        # parser.add_argument('--backmodel_hparams_file2', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')

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
#         # casual_mask = self.casual_mask[:, :, :t, :t]
#         # if tgt_mask is not None:
#         #     pad_future_mask = casual_mask & tgt_mask
#         # else:
#         #     pad_future_mask = casual_mask
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