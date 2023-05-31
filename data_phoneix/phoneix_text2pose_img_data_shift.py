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
from util.plot_videos import draw_frame_2D

# POSE_MAX_X = 1280
# POSE_MAX_Y = 720
# POSE_MIN_X = -1280
# POSE_MIN_Y = -720


class PoseDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, opts, mode):
        """
        Args:
            csv_path: Data/
            data_folder: "/Dataset/how2sign/video_and_keypoint/"
            keypoint_folder: 
            video_folder: /Dataset/how2sign/video_and_keypoint/train/videos
        """
        super().__init__()
        # self.hand_generator = opts.hand_generator
        data_path = opts.data_path
        self.max_frame_len = opts.max_frames_num

        tag = mode
        self.tag = tag
        
        gloss_file = os.path.join(opts.data_path, "{}.gloss".format(tag))
        skel_file = os.path.join(opts.data_path, "{}.skels".format(tag))
        file_path = os.path.join(opts.data_path, "{}.files".format(tag))

        text_dict = Dictionary()

        self.text_dict = text_dict
        text_dict = text_dict.load(opts.vocab_file)

        self.gloss = []
        self.gloss_id = []
        self.skel_3d = []
        self.skel_len = []
        self.gloss_len = []
        self.vid_path = []


        with open(gloss_file, "r") as f1, open(skel_file, "r") as f2, open(file_path, "r") as f3:
            for i, (gloss_line, skel_line, file_line) in enumerate(zip(f1, f2, f3)):
                # if i > 100: break
                gloss = gloss_line.strip()
                gloss_ids = text_dict.encode_line(gloss)
                gloss_len = len(gloss_ids)
                skels_3d = torch.FloatTensor([float(s) for s in skel_line.strip().split()])

                vid_path = os.path.join("/Dataset/everybody_sign_now_experiments/images", file_line.strip())

                assert len(skels_3d) % 151 == 0
                skel_len = len(skels_3d) // 151

                self.gloss.append(gloss)
                self.gloss_id.append(gloss_ids)         
                self.skel_3d.append(skels_3d)         
                self.skel_len.append(skel_len)  
                self.gloss_len.append(gloss_len) 
                self.vid_path.append(vid_path) 

        self.input_shape = 128
        self.transform = transforms.Compose([
            transforms.Resize((self.input_shape, self.input_shape)),
        ])


    def __len__(self):
        return len(self.gloss)

    def __getitem__(self, idx):
        gloss = self.gloss[idx]
        gloss_id = self.gloss_id[idx]
        gloss_len = self.gloss_len[idx]

        skel_3d = self.skel_3d[idx]
        skel_len = self.skel_len[idx]

        skel_3d_2d = skel_3d.reshape(skel_len, 151)
        
        imgs = []
        for i in range(skel_len):
            cur_frame = skel_3d_2d[i][:-1]  # [150]
            frame = np.ones((256, 256, 3), np.uint8) * 255
            frame_joints_2d = np.reshape(cur_frame, (50, 3))[:, :2]
            # Draw the frame given 2D joints
            im = draw_frame_2D(frame, frame_joints_2d) # [h, w, c]
            im = torch.FloatTensor(im).permute(2,0,1).contiguous() # [c, h, w]
            im = self.transform(im)
            imgs.append(im.unsqueeze(0))
        imgs = torch.cat(imgs, dim=0)
        return dict(gloss=gloss, gloss_id=gloss_id, skel_3d=skel_3d, gloss_len=gloss_len, skel_len=skel_len, vid=imgs)


    def get_images(self, video_name):
        frames_list = glob.glob(os.path.join(video_name, '*.{}'.format("png")))
        frames_list.sort()
        num_frame = len(frames_list)
        return frames_list

    def load_video_from_images(self, frames_list):
        frames_tensor_list = [self.load_image(frame_file, self.tag) for frame_file in frames_list]
        video_tensor = torch.stack(frames_tensor_list, dim=0)
        return video_tensor

    def load_image(self, img_name, phase, reduce_mean=True):
        image = Image.open(img_name)
        image = self.transform(image)
        return image

    def collate_fn(self, batch):        
        gloss_id = self.collate_points([x["gloss_id"] for x in batch], pad_idx=self.text_dict.pad())
        skel_3d = self.collate_points([x["skel_3d"] for x in batch], pad_idx=0.)

        bs = gloss_id.size(0)
        skel_3d = skel_3d.view(bs, -1, 151)[:, :, :-1].contiguous() # [bs, max_len, 150]

        gloss = [x["gloss"] for x in batch]
        skel_len = torch.LongTensor([x["skel_len"] for x in batch])
        gloss_len = torch.LongTensor([x["gloss_len"] for x in batch])

        max_len = max(skel_len).item()

        vids = [x["vid"] for x in batch]
        video = torch.zeros(bs, max_len, 3, self.input_shape, self.input_shape)
        for i in range(bs):
            video[i, :skel_len[i].item(), ...] = vids[i]

        video /= 255.
        return dict(gloss=gloss, gloss_id=gloss_id.long(), skel_3d=skel_3d, skel_len=skel_len, gloss_len=gloss_len, vid=video)


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

    def _dataloader(self, mode):
        dataset = self._dataset(mode)
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
            pin_memory=False,
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

    # dataloader = PhoenixPoseData(opts).val_dataloader()
    dataloader = PoseDataset(opts, "dev")

    print(len(dataloader))
    for i, data in enumerate(dataloader):
        if i > 5: break
        print("")
        print("gloss: ", data["gloss"])
        print("gloss_id: ", data["gloss_id"])
        print("skel_id: ", data["skel_3d"].shape) # [0, 0, 5:, :]
        print("skel_len: ", data["skel_len"])
        print("video: ", data["vid"].shape)
        # print("pose: ", data["pose"].shape, data["pose"][:, :2, 4], data["pose"][:, :2, 7])
        # print("rhand: ", data["rhand"].shape, data["rhand"][:, :2, 0])
        # print("lhand: ", data["lhand"].shape, data["lhand"][:, :2, 0])
    # exit()
