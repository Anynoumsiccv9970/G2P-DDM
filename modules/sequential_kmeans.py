
from cProfile import label
import einops
from configs.train_options import TrainOptions
import pytorch_lightning as pl
import argparse
import numpy as np
from models_phoneix.point2text_model_vqvae_tr_nat_stage1_seperate2 import Point2textModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from data_phoneix.stage1_phoneix_data import PhoenixPoseData, PoseDataset
from util.util import CheckpointEveryNSteps
import os
from data.vocabulary import Dictionary
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from .density_peak_cluster import knn_dpc
import math
from tqdm import tqdm
from collections import defaultdict


pl.seed_everything(1234)
parser = argparse.ArgumentParser()
parser = Point2textModel.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
opt = TrainOptions(parser).parse()

opt.data_path = "Data/ProgressiveTransformersSLP"
opt.vocab_file = "Data/ProgressiveTransformersSLP/src_vocab.txt"
opt.batchSize = 5

data = PhoenixPoseData(opt)
train_data = data.train_dataloader()
val_data = data.val_dataloader()
test_data = data.test_dataloader()

text_dict = Dictionary()
text_dict = text_dict.load(opt.vocab_file)

ckpt_path = "/Dataset/everybody_sign_now_experiments/pose2text_logs/stage1/lightning_logs/seperate_vit/checkpoints/epoch=56-step=33743-val_wer=0.0000-val_rec_loss=0.0138-val_ce_loss=0.0000.ckpt"
ckpt_hparams_file = "/Dataset/everybody_sign_now_experiments/pose2text_logs/stage1/lightning_logs/seperate_vit/hparams.yaml"
model = Point2textModel.load_from_checkpoint(ckpt_path, hparams_file=ckpt_hparams_file)
mode = model.cuda()

def mask_non_local_mask(size, local_ws=16):
    """
    :param size:
    :param local_ws:
    :return:
    tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]])
    """
    tmp = torch.ones(size, size).long()
    mask = torch.triu(tmp, diagonal=int(local_ws/2)) | (1 - torch.triu(tmp, diagonal=-int(local_ws/2-1)))
    return (1 - mask)


def get_elu_dis(x1, x2):
    dis = ((x1 ** 2) - 2 * x1 * x2 + x2 ** 2).sum(dim=1)
    return dis


with torch.no_grad():
    total_pred_leng = []
    total_gloss = []
    for batch_idx, batch in tqdm(enumerate(val_data)):
        gloss = batch["gloss"]   # [bs, src_len]
        gloss_id = batch["gloss_id"].cuda()   # [bs, src_len]
        gloss_len = batch["gloss_len"].cuda() # list(src_len)
        points = batch["skel_3d"].cuda()     # [bs, max_len, 150]
        skel_len = batch["skel_len"].cuda()   # list(skel_len)
        bs, max_len, v = points.size()

        max_len = max(skel_len)
        points_mask = model._get_mask(skel_len, max_len, points.device)
        # vqvae encoder
        _, points_feat, _ = model.vqvae_encode(points, points_mask)

        points_feat = einops.rearrange(points_feat, "b h t n -> b t (h n)")

        n_clusters = gloss_len
        for i in range(bs):
            n_cluster = n_clusters[i].item()
            cur_leng = skel_len[i].item()
            feats = points_feat[i, :cur_leng]
            # print("n_cluster: ", n_cluster)
            kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(feats.cpu().numpy())
            kmean_label = kmeans.labels_
            
            # print("kmean_label: ", len(kmean_label), kmean_label)
            localdens = []
            pre_lab = kmean_label[0]
            cur_idx = [0]
            for lab_idx in range(1, len(kmean_label)):
                if kmean_label[lab_idx] == pre_lab:
                    cur_idx.append(lab_idx)
                    if lab_idx == len(kmean_label) - 1:
                        localdens.append(cur_idx)
                else:
                    localdens.append(cur_idx)
                    pre_lab = kmean_label[lab_idx]
                    cur_idx = [lab_idx]

            rho = {}
            for lnum in localdens:
                rho[math.floor(np.mean(lnum))] = len(lnum)
            sort_rho = sorted(rho.items(), key=lambda item: item[1], reverse=True)
            peaks = sorted([x[0] for x in sort_rho[:n_cluster]])
            # print("peaks : ", peaks, kmean_label[peaks])
            
            if len(peaks) != n_cluster:
                print(kmean_label, localdens, rho, len(peaks), n_cluster)
                print("peak is not equal: ", gloss[i])
            assert len(peaks) == n_cluster

            
            boundaries = [0]
            boundaries = [0]
            for p in range(len(peaks)-1):
                pre = peaks[p]
                post = peaks[p+1]
                middle = kmean_label[pre + 1 : post+1]
                for m in range(len(middle)):
                    if middle[m] != kmean_label[pre]:
                        boundaries.append(pre + 1 + m)
                        break
            
            boundaries.append(cur_leng)
            # print("boundaries: ", boundaries)

            lengths = [boundaries[i] - boundaries[i-1] for i in range(1, len(boundaries))]
            # print("lengths: ", lengths, len(lengths), n_cluster)
            if len(lengths) != n_cluster:
                print("lengths is not equal: ", gloss[i])
            assert sum(lengths) == cur_leng
            total_pred_leng.append(lengths)

print("total_pred_leng: ", len(total_pred_leng))
with open("Data/ProgressiveTransformersSLP/dev.leng", "w") as f:
    for lengs in total_pred_leng:
        lengs = " ".join([str(x) for x in lengs])
        f.write(lengs + "\n")

