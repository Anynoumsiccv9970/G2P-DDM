import cv2 as cv 
import numpy as np
import scipy
import math
import time
import copy
import matplotlib
#%matplotlib inline
import pylab as plt
import json
from PIL import Image
from shutil import copyfile
from skimage import img_as_float
from functools import reduce
from .renderopenpose import *
import os
import argparse

myshape = (720, 1280, 3)


def readkeypointsfile_json(myfile):
    f = open(myfile, 'r')
    json_dict = json.load(f)
    people = json_dict['people']
    posepts =[]
    facepts = []
    r_handpts = []
    l_handpts = []
    for p in people:
        posepts += p['pose_keypoints_2d']
        facepts += p['face_keypoints_2d']
        r_handpts += p['hand_right_keypoints_2d']
        l_handpts += p['hand_left_keypoints_2d']
    return posepts, facepts, r_handpts, l_handpts

path = "/Dataset/how2sign/video_and_keypoint/train/openpose_output/json/CJO5S96W7Cs_0-8-rgb_front/CJO5S96W7Cs_0-8-rgb_front_000000000009_keypoints.json"

posepts, facepts, r_handpts, l_handpts = readkeypointsfile_json(path)

print(len(posepts))

canvas = renderpose(posepts, 255 * np.ones(myshape, dtype='uint8'))
canvas = renderface_sparse(facepts, canvas, 70, disp=False)
canvas = renderhand(r_handpts, canvas)
canvas = renderhand(l_handpts, canvas)

canvas = Image.fromarray(canvas[:, :, [2,1,0]])

canvas.save("test" + '.png')