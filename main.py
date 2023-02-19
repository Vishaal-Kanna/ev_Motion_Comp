#!/usr/bin/evn python

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import cv2
import os
import argparse
from utils.ev_to_voxels import events_to_neg_pos_ts_voxel_torch, events_to_neg_pos_n_voxel_torch
from utils.warp_functions import pose2flow, flow_warp
from Optimization.opt_layer import Optimization

h = 180
w = 240

def main():
    device = torch.device("cuda")

    frames = np.loadtxt('./Sample_data/test_data.txt')

    frames = frames.T
    xs = torch.from_numpy(frames[0])
    ys = torch.from_numpy(frames[1])
    ts = torch.from_numpy(frames[2])
    ps = torch.from_numpy(frames[3])

    _, _, dT = events_to_neg_pos_ts_voxel_torch(xs, ys, ts, ps, 9)
    img_n_test, _ = events_to_neg_pos_n_voxel_torch(xs, ys, ts, ps, 9)
    img_n_test = torch.from_numpy(np.transpose(img_n_test, axes=(1, 0, 2, 3))).float()

    tensor = []
    tensor.append(img_n_test)
    batch_tensor_test = (torch.stack(tensor, 0)).to(device)

    Opt_layer = Optimization().to(device)

    img_after_comp = Opt_layer(batch_tensor_test, dT)

if __name__ == '__main__':
    main()
