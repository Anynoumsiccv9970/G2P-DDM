from genericpath import exists
import os
import os.path as osp
import math
import random
import pickle
from re import L
import warnings

import glob
import h5py
import numpy as np

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
import cv2
import pandas as pd
from tqdm import tqdm
from torchvision.datasets.video_utils import VideoClips
import json
from PIL import Image
import torchvision.transforms as transforms
from data.vocabulary import Dictionary

# POSE_MAX_X = 1280
# POSE_MAX_Y = 720
# POSE_MIN_X = -1280
# POSE_MIN_Y = -720


class PoseDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, opts, mode="train"):
        """
        Args:
            csv_path: Data/
            data_folder: "/Dataset/how2sign/video_and_keypoint/"
            keypoint_folder: 
            video_folder: /Dataset/how2sign/video_and_keypoint/train/videos
        """
        super().__init__()
        self.train = mode
        # self.hand_generator = opts.hand_generator
        data_path = opts.data_path
        max_frame_len = 300

        tag = mode
        
        gloss_file = os.path.join(opts.data_path, "{}.gloss".format(tag))
        skel_file = os.path.join(opts.data_path, "{}.skels".format(tag))
        leng_file = os.path.join(opts.data_path, "{}.leng".format(tag))

        text_dict = Dictionary()

        self.text_dict = text_dict
        text_dict = text_dict.load(opts.vocab_file)

        self.gloss = []
        self.pred_leng = []
        self.gloss_id = []
        self.skel_3d = []
        self.skel_len = []
        self.gloss_len = []
        with open(gloss_file, "r") as f1, open(skel_file, "r") as f2, open(leng_file, "r") as f3:
            for i, (gloss_line, skel_line, leng_line) in enumerate(zip(f1, f2, f3)):
                # if i > 100: break
                gloss = gloss_line.strip()
                gloss_ids = text_dict.encode_line(gloss, append_eos=False)
                pred_leng = [int(x) for x in leng_line.strip().split()]
                gloss_len = len(gloss_ids)
                skels_3d = torch.FloatTensor([float(s) for s in skel_line.strip().split()])
                # print("len(skels_3d): ", len(skels_3d))

                assert len(skels_3d) % 151 == 0
                skel_len = len(skels_3d) // 151
                

                self.gloss.append(gloss)
                self.gloss_id.append(gloss_ids)         
                self.skel_3d.append(skels_3d)
                self.skel_len.append(skel_len)  
                self.gloss_len.append(gloss_len)
                self.pred_leng.append(pred_leng)       

    def __len__(self):
        return len(self.gloss)

    def __getitem__(self, idx):
        gloss = self.gloss[idx]
        pred_len = self.pred_leng[idx]
        gloss_id = self.gloss_id[idx]
        gloss_len = self.gloss_len[idx]

        skel_3d = self.skel_3d[idx]
        skel_len = self.skel_len[idx]
        
        skel_len = skel_len # // 4 * 4
        skel_3d = skel_3d[:skel_len * 151]
        
        return dict(gloss=gloss, pred_len=pred_len, gloss_id=gloss_id, skel_3d=skel_3d, gloss_len=gloss_len, skel_len=skel_len)

    def collate_fn(self, batch):        
        gloss_id = self.collate_points([x["gloss_id"] for x in batch], pad_idx=self.text_dict.pad())
        skel_3d = self.collate_points([x["skel_3d"] for x in batch], pad_idx=0.)

        bs, max_len = skel_3d.size()
        assert max_len % 151 ==0
        skel_3d = skel_3d.view(bs, -1, 151)[:, :, :-1].contiguous() # [bs, max_len, 150]

        gloss = [x["gloss"] for x in batch]
        pred_len = [x["pred_len"] for x in batch]
        skel_len = torch.LongTensor([x["skel_len"] for x in batch])
        gloss_len = torch.LongTensor([x["gloss_len"] for x in batch])
        return dict(gloss=gloss, pred_len=pred_len, gloss_id=gloss_id.long(), skel_3d=skel_3d, skel_len=skel_len, gloss_len=gloss_len)


    def collate_points(self, values, pad_idx, left_pad=False):
        
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)
        
        for i, v in enumerate(values):
            copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        return res


    

class PhoenixPoseData(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        
    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes

    def _dataset(self, mode):
        Dataset = PoseDataset
        dataset = Dataset(self.args, mode=mode)
        return dataset

    def _dataloader(self, train):
        dataset = self._dataset(train)
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.args.batchSize,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader("train")

    def val_dataloader(self):
        return self._dataloader("dev")

    def test_dataloader(self):
        return self._dataloader("test")


if __name__ == "__main__":
    pass
    csv_path = "Data"
    data_path = "Data/ProgressiveTransformersSLP/"
    vocab_file = data_path + "src_vocab.txt"
    class Option:
        vocab_file = vocab_file
        hand_generator = True
        resolution = 256
        csv_path=csv_path
        data_path=data_path
        batchSize=2
        num_workers=32
        sequence_length=8
        debug = 100
    opts= Option()

    dataloader = PhoenixPoseData(opts).val_dataloader()
    # dataloader = PoseDataset(opts, False)
    print(len(dataloader))
    for i, data in enumerate(dataloader):
        if i > 0: break
        print("")
        print("gloss: ", data["gloss"])
        print("gloss_id: ", data["gloss_id"])
        print("skel_id: ", data["skel_3d"].shape, data["skel_3d"]) # [0, 0, 5:, :]
        print("skel_len: ", data["skel_len"])
        # print("pose: ", data["pose"].shape, data["pose"][:, :2, 4], data["pose"][:, :2, 7])
        # print("rhand: ", data["rhand"].shape, data["rhand"][:, :2, 0])
        # print("lhand: ", data["lhand"].shape, data["lhand"][:, :2, 0])
    # exit()
