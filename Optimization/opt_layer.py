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

h = 180
w = 240

class Optimization(nn.Module):
    def __init__(self):
        super(Optimization, self).__init__()

    def forward(self, input_n, dT):
        """
        Apply motion compensation for voxelized event representation
        @param input_n event voxels
        @param dT time interval between thestarting and ending events
        @returns Motion compensated event image
        """

        cf = 100
        itr = 1
        lr = 0.005
        m_model = Variable(torch.ones(4), requires_grad=True).cuda()
        while(itr<=500):

            cf, warped_image = self.Cost(input_n, dT, m_model)
            # print(itr, ' : ', -cf.item())
            m_model.retain_grad()
            cf.backward(retain_graph=True)
            m_model = m_model - (lr * m_model.grad.data)

            im_target = warped_image.data.cpu().numpy().reshape((h, w))

            im_target = im_target.reshape((1, h, w))
            im = np.transpose(im_target, axes=(1, 2, 0))

            if itr==500:
                cv2.imshow("After Compensation", im)
                cv2.waitKey(0)
                return im
            if itr == 1:
                cv2.imshow("Before Compensation", im)
                cv2.waitKey(0)

            itr+=1
        return m_model, cf

    def Cost(self, input_n, dT, mmodel):

        dt = dT/9

        device = torch.device("cuda")

        # bin0
        flow = pose2flow(mmodel, 0 * dt, device = device)
        cluster_i_IWE_forward_bin0_n = flow_warp(input_n[:, :, 0, :, :], flow)

        # bin1
        flow = pose2flow(mmodel, (-1) * dt, device = device)
        cluster_i_IWE_forward_bin1_n = flow_warp(input_n[:, :, 1, :, :], flow)

        # bin2
        flow = pose2flow(mmodel, (-2) * dt, device = device)
        cluster_i_IWE_forward_bin2_n = flow_warp(input_n[:, :, 2, :, :], flow)

        # bin3
        flow = pose2flow(mmodel, (-3) * dt, device = device)
        cluster_i_IWE_forward_bin3_n = flow_warp(input_n[:, :, 3, :, :], flow)

        # bin4
        flow = pose2flow(mmodel, (-4) * dt, device = device)
        cluster_i_IWE_forward_bin4_n = flow_warp(input_n[:, :, 4, :, :], flow)

        # bin5
        flow = pose2flow(mmodel, (-5) * dt, device = device)
        cluster_i_IWE_forward_bin5_n = flow_warp(input_n[:, :, 5, :, :], flow)

        # bin6
        flow = pose2flow(mmodel, (-6) * dt, device = device)
        cluster_i_IWE_forward_bin6_n = flow_warp(input_n[:, :, 6, :, :], flow)

        # bin7
        flow = pose2flow(mmodel, (-7) * dt, device = device)
        cluster_i_IWE_forward_bin7_n = flow_warp(input_n[:, :, 7, :, :], flow)

        # bin8
        flow = pose2flow(mmodel, (-8) * dt, device = device)
        cluster_i_IWE_forward_bin8_n = flow_warp(input_n[:, :, 8, :, :], flow)

        iwe_j = torch.add(cluster_i_IWE_forward_bin0_n, cluster_i_IWE_forward_bin1_n)
        iwe_j = torch.add(iwe_j, cluster_i_IWE_forward_bin2_n)
        iwe_j = torch.add(iwe_j, cluster_i_IWE_forward_bin3_n)
        iwe_j = torch.add(iwe_j, cluster_i_IWE_forward_bin4_n)
        iwe_j = torch.add(iwe_j, cluster_i_IWE_forward_bin5_n)
        iwe_j = torch.add(iwe_j, cluster_i_IWE_forward_bin6_n)
        iwe_j = torch.add(iwe_j, cluster_i_IWE_forward_bin7_n)
        iwe_j = torch.add(iwe_j, cluster_i_IWE_forward_bin8_n)

        iwe = iwe_j[:,0,:,:] + iwe_j[:,1,:,:]
        # Cost Function
        var_iwe_j = torch.var(iwe_j[:,0,:,:] - torch.mean(iwe_j[:,0,:,:])) + torch.var(iwe_j[:,1,:,:] - torch.mean(iwe_j[:,1,:,:]))

        return -var_iwe_j, iwe
