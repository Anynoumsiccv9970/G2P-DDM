from email.policy import default
from turtle import forward
from matplotlib.pyplot import text
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
import torchvision.transforms as transforms
from util.dtw import calculate_dtw, dtw





class Point2textModel(pl.LightningModule):
    def __init__(self, args, text_dict):
        super().__init__()
        self.args = args
        self.text_dict = text_dict
        # vqvae
        self.tem_pos_emb = nn.Parameter(torch.zeros(2000, 256))
        self.spa_pos_emb = nn.Parameter(torch.zeros(3, 256))
        self.tem_pos_emb.data.normal_(0, 0.02)
        self.spa_pos_emb.data.normal_(0, 0.02)

        # self.points_emb = nn.Linear(150, 256)
        self.pose_emb = nn.Linear(24, 256)
        self.rhand_emb = nn.Linear(63, 256)
        self.lhand_emb = nn.Linear(63, 256)

        self.enc_tem_vit = Encoder(dim=256, depth=3, heads=8, mlp_dim=1024, dropout = 0.1)
        self.enc_spa_vit = Encoder(dim=256, depth=3, heads=8, mlp_dim=1024, dropout = 0.1)
        
        self.codebook = Codebook(n_codes=args.n_codes, embedding_dim=256)
        self.dec_tem_vit = Encoder(dim=256, depth=3, heads=8, mlp_dim=1024, dropout = 0.1)
        self.dec_spa_vit = Encoder(dim=256, depth=3, heads=8, mlp_dim=1024, dropout = 0.1)
        self.dec_linear = nn.Linear(256, 256)
        self.pose_spl = SPL(input_size=256, hidden_layers=5, hidden_units=256, joint_size=3, reuse=False, sparse=False, SKELETON="sign_pose")        
        self.hand_spl = SPL(input_size=256, hidden_layers=5, hidden_units=256, joint_size=3, reuse=False, sparse=False, SKELETON="sign_hand")

        from models_phoneix.point2text_model_cnn import BackTranslateModel as BackTranslateModel1
        from models_phoneix.point2text_model import BackTranslateModel as BackTranslateModel2
        self.back_translate_model1 = BackTranslateModel1.load_from_checkpoint(
            checkpoint_path="/Dataset/everybody_sign_now_experiments/pose2text_logs/backmodel/lightning_logs/heatmap_model/checkpoints/epoch=7-step=28383-val_wer=0.6861.ckpt", 
            hparams_file="/Dataset/everybody_sign_now_experiments/pose2text_logs/backmodel/lightning_logs/heatmap_model/hparams.yaml",
            strict=False)
        self.back_translate_model2 = BackTranslateModel2.load_from_checkpoint(
            checkpoint_path="/Dataset/everybody_sign_now_experiments/pose2text_logs/backmodel/lightning_logs/joint_model/checkpoints/epoch=13-step=8287-val_wer=0.5971.ckpt",
            hparams_file="/Dataset/everybody_sign_now_experiments/pose2text_logs/backmodel/lightning_logs/joint_model/hparams.yaml",
            strict=False)

        self.save_hyperparameters()


    def vqvae_encode(self, points, points_mask):
        """points: [b t h]
           points_mask: [b t]
        """
        pose = points[:, :, :24]
        rhand = points[:, :, 24:24+63]
        lhand = points[:, :, 87:150]
        b = pose.size(0)

        pose = self.pose_emb(pose).unsqueeze_(-2) # [bs, t, 1, 256]
        rhand = self.rhand_emb(rhand).unsqueeze_(-2) # [bs, t, 1, 256]
        lhand = self.lhand_emb(lhand).unsqueeze_(-2) # [bs, t, 1, 256]
        points_feat = torch.cat([pose, rhand, lhand], dim=-2) # [bs, t, 3, 256]

        # spatial vit
        points_feat = einops.rearrange(points_feat, "b t n h -> (b t) n h")
        points_feat = points_feat + self.spa_pos_emb[:points_feat.size(1), :] # TODO
        points_feat = self.enc_spa_vit(points_feat, mask=None)

        # temporal vit
        points_feat = einops.rearrange(points_feat, "(b t) n h -> (b n) t h", b=b)
        points_feat = points_feat + self.tem_pos_emb[:points_feat.size(1), :] # TODO
        points_mask = einops.repeat(points_mask, "b t-> (b n) t", n=3)
        points_feat = self.enc_tem_vit(points_feat, points_mask.unsqueeze_(1).unsqueeze_(2)) # [bs, max_len, 256]
        points_feat = einops.rearrange(points_feat, "(b n) t h -> b t n h", b=b, n=3)
        
        # codebook
        points_feat = einops.rearrange(points_feat, "b t n h -> b h t n")
        vq_output = self.codebook(points_feat)
        vq_tokens, points_feat, commitment_loss = vq_output['encodings'], vq_output['embeddings'], vq_output["commitment_loss"] # [bs, max_len, 3]
        return vq_tokens, points_feat, commitment_loss

    def vqvae_decode(self, points_feat, points_mask):
        """points_feat: [b h t n]
        """
        b = points_feat.size(0)
        points_feat = einops.rearrange(points_feat, "b h t n-> (b t) n h")      # [bs, max_len, 3, h]
        points_feat = points_feat + self.spa_pos_emb[:points_feat.size(1), :] # TODO
        points_feat = self.dec_spa_vit(points_feat, mask=None)

        points_feat = einops.rearrange(points_feat, "(b t) n h-> (b n) t h", b=b)      # [bs*3, max_len, h]
        points_feat = points_feat + self.tem_pos_emb[:points_feat.size(1), :] # TODO
        points_mask = einops.repeat(points_mask, "b t-> (b n) t", n=3)
        points_feat = self.dec_tem_vit(points_feat, points_mask.unsqueeze(1).unsqueeze(2))  # [b*3, max_len h]
        
        # reconstruction loss
        rec_feat = einops.rearrange(points_feat, "(b n) t h -> b t n h", b=b, n=3) # [bs, t, 3, h]
        dec_pose = rec_feat[:, :, 0, :]  # [bs, t, h]
        dec_pose = einops.rearrange(dec_pose, "b t h -> (b t) h")
        dec_rhand = rec_feat[:, :, 1, :] # [bs, t, h]
        dec_rhand = einops.rearrange(dec_rhand, "b t h -> (b t) h")
        dec_lhand = rec_feat[:, :, 2, :] # [bs, t, h]
        dec_lhand = einops.rearrange(dec_lhand, "b t h -> (b t) h")

        dec_pose = self.pose_spl(dec_pose)  # [b, h] -> [b, 24]
        dec_rhand = self.hand_spl(dec_rhand) # [b, h] -> [b, 63]
        dec_lhand = self.hand_spl(dec_lhand) # [b, h] -> [b, 63]

        return dec_pose, dec_rhand, dec_lhand


    def forward(self, batch, mode, vis_func, vis_tok_func):
        """[bs, t, 150]
        """
        gloss_id = batch["gloss_id"]   # [bs, src_len]
        gloss_len = batch["gloss_len"] # list(src_len)
        points = batch["skel_3d"]      # [bs, max_len, 150]
        skel_len = batch["skel_len"]   # list(skel_len)
        bs, max_len, v = points.size()

        max_len = max(skel_len)
        points_mask = self._get_mask(skel_len, max_len, points.device)
        # vqvae encoder
        vq_tokens, points_feat, commitment_loss = self.vqvae_encode(points, points_mask) # [bs, t], [bs, h, t]
        self.log('{}/commitment_loss'.format(mode), commitment_loss.detach(), prog_bar=True)
        # vqvae decoder
        dec_pose, dec_rhand, dec_lhand = self.vqvae_decode(points_feat, points_mask)

        dec_points = torch.cat([dec_pose, dec_rhand, dec_lhand], dim=-1) # [bs*max_len, 150]

        ori_points = einops.rearrange(points, "b t v -> (b t) v")
        pose = ori_points[:, :24]
        rhand = ori_points[:, 24:24+63]
        lhand = ori_points[:, 87:150]

        ori_mask = self._get_mask(skel_len, max_len, points.device)
        rec_loss = torch.abs(dec_points - ori_points)[ori_mask.view(-1)].mean()
        self.log('{}_rec_loss'.format(mode), rec_loss.detach(), prog_bar=True)

        if mode == "train" and self.global_step % 500 == 0:
            vis_func(dec_pose, dec_rhand, dec_lhand, mode, "recons", vis_len=skel_len[0].item())
            vis_func(pose, rhand, lhand, mode, "origin", vis_len=skel_len[0].item())
            vis_tok_func(vq_tokens[0, :skel_len[0].item()], mode, "rec")
        elif mode == "val":
            vis_func(dec_pose, dec_rhand, dec_lhand, mode, "recons", vis_len=skel_len[0].item())
            vis_func(pose, rhand, lhand, mode, "origin", vis_len=skel_len[0].item())
            vis_tok_func(vq_tokens[0, :skel_len[0].item()], mode, "rec")
        
        loss = rec_loss + commitment_loss
        self.log('{}/loss'.format(mode), loss.detach(), prog_bar=True)
        return loss, dec_points, ori_points


    def training_step(self, batch):
        loss, _, _ = self.forward(batch, "train", self.vis, self.vis_token)
        return loss

    
    def validation_step(self, batch, batch_idx):
        gloss_id = batch["gloss_id"]  # [bs, src_len]
        gloss_len = batch["gloss_len"] # list(src_len)
        points = batch["skel_3d"]      # [bs, max_len, 150]
        skel_len = batch["skel_len"]   # list(skel_len)
        _, dec_points, ori_points = self.forward(batch, "val", self.vis, self.vis_token)

        bs, max_len, _ = points.size()
        # dec_points = einops.rearrange(dec_points, "(b t) v -> b t v", b=bs, t=max_len)
        # # dec_video = self.points2imgs(dec_points, skel_len)
        # # rec_res1 = self._compute_wer(dec_video, skel_len, gloss_id, gloss_len, "test", self.back_translate_model1)
        # rec_res2 = self._compute_wer(dec_points, skel_len, gloss_id, gloss_len, "test", self.back_translate_model2)
        # rec_res1 = rec_res2

        # ori_points = einops.rearrange(ori_points, "(b t) v -> b t v", b=bs, t=max_len)
        # # ori_video = self.points2imgs(ori_points, skel_len)
        # # ori_res1 = self._compute_wer(ori_video, skel_len, gloss_id, gloss_len, "test", self.back_translate_model1)
        # ori_res2 = self._compute_wer(ori_points, skel_len, gloss_id, gloss_len, "test", self.back_translate_model2)
        # ori_res1 = ori_res2

        # dtw_scores = []
        # for i in range(bs):
        #     dec_point = dec_points[i, :skel_len[i].item(), :].cpu().numpy()
        #     ori_point = ori_points[i, :skel_len[i].item(), :].cpu().numpy()
            
        #     euclidean_norm = lambda x, y: np.sum(np.abs(x - y))
        #     d, cost_matrix, acc_cost_matrix, path = dtw(dec_point, ori_point, dist=euclidean_norm)

        #     # Normalise the dtw cost by sequence length
        #     dtw_scores.append(d/acc_cost_matrix.shape[0])
        
        # return rec_res1, ori_res1, dtw_scores, rec_res2, ori_res2


    # def validation_epoch_end(self, outputs) -> None:
    #     rec_err, rec_correct, rec_count = np.zeros([4]), 0, 0
    #     ori_err, ori_correct, ori_count = np.zeros([4]), 0, 0
    #     rec_err2, rec_correct2, rec_count2 = np.zeros([4]), 0, 0
    #     ori_err2, ori_correct2, ori_count2 = np.zeros([4]), 0, 0
    #     dtw_scores = []
    #     for rec_out, ori_out, dtw, rec_out2, ori_out2, in outputs:
    #         rec_err += rec_out["wer"]
    #         rec_correct += rec_out["correct"]
    #         rec_count += rec_out["count"]
    #         ori_err += ori_out["wer"]
    #         ori_correct += ori_out["correct"]
    #         ori_count += ori_out["count"]
            
    #         dtw_scores.extend(dtw)
            
    #         rec_err2 += rec_out2["wer"]
    #         rec_correct2 += rec_out2["correct"]
    #         rec_count2 += rec_out2["count"]
    #         ori_err2 += ori_out2["wer"]
    #         ori_correct2 += ori_out2["correct"]
    #         ori_count2 += ori_out2["count"]

    #     self.log('{}/acc'.format("rec"), rec_correct / rec_count, prog_bar=True)
    #     self.log('{}_wer'.format("rec"), rec_err[0] / rec_count, prog_bar=True)
    #     self.log('{}/acc'.format("ori"), ori_correct / ori_count, prog_bar=True)
    #     self.log('{}_wer'.format("ori"), ori_err[0] / ori_count, prog_bar=True)
        
    #     self.log('{}_dtw'.format("test"), sum(dtw_scores) / len(dtw_scores), prog_bar=True)

    #     self.log('{}/acc2'.format("rec2"), rec_correct2 / rec_count2, prog_bar=True)
    #     self.log('{}_wer2'.format("rec2"), rec_err2[0] / rec_count2, prog_bar=True)
    #     self.log('{}/acc2'.format("ori2"), ori_correct2 / ori_count2, prog_bar=True)
    #     self.log('{}_wer2'.format("ori2"), ori_err2[0] / ori_count2, prog_bar=True)

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
            frame = np.ones((256, 256, 3), np.uint8) * 255
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
        parser.add_argument('--vqvae_hparams_file', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--embedding_dim', type=int, default=256)
        parser.add_argument('--n_codes', type=int, default=1024)
        parser.add_argument('--n_hiddens', type=int, default=256)
        parser.add_argument('--n_res_layers', type=int, default=2)
        parser.add_argument('--downsample', nargs='+', type=int, default=(4, 4, 4))
        parser.add_argument('--backmodel', type=str, default='kinetics_stride4x4x4', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--backmodel_hparams_file', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--backmodel2', type=str, default='kinetics_stride4x4x4', help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--backmodel_hparams_file2', type=str, default='', help='path to vqvae ckpt, or model name to download pretrained')
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
            emb = emb + self.pe[:emb.size(1)]
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
    def __init__(self, emb_dim=256, depth=6, block_size=2000):
        super().__init__()
        casual_mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("casual_mask", casual_mask.bool().view(1, 1, block_size, block_size))

        self.encoder = Encoder(dim=emb_dim, depth=depth, heads=8, mlp_dim=1024, dropout = 0.1)
        self.decoder = Decoder(dim=emb_dim, depth=depth, heads=8, mlp_dim=1024, dropout = 0.1)


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