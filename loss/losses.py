# -*- coding: utf-8 -*-
# @Author  : jingyi

import torch
import torch.nn as nn
from loss.chamfer_loss import ChamferLoss
from loss.nn_loss import NN_loss
from loss.pose_loss import Pose_Loss


class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.chamferloss = ChamferLoss()
        self.nnloss = NN_loss()
        self.l1loss = nn.L1Loss()
        self.poseloss = Pose_Loss(cfg.LOSS.zmp_loss, cfg.LOSS.bone_loss)

    def forward(self, output):
        # pred = output['predict_normal_pc']
        # target = output['sample_pc']
        total_loss = 0
        total_losses = {}
        others = {}

        if self.cfg.LOSS.chamfer_loss:
            chamfer_loss = self.chamferloss(output['predict_normal_pc'],output['sample_pc'])
            total_losses.update(dict(chamfer_loss=chamfer_loss))

        if self.cfg.LOSS.nn_loss:
            nn_loss = self.nnloss(output['predict_normal_pc'], output['sample_pc'])
            total_losses.update(dict(nn_loss=nn_loss.mean()))
            
        if self.cfg.LOSS.flow_loss:
            l1_loss = self.l1loss(output['flow'], output['flow_truth'])
            total_losses.update(dict(l1_loss=l1_loss))

        if self.cfg.LOSS.end_loss:
            end_loss = self.l1loss(output['predict_normal_pc'], output['sample_pc'])
            total_losses.update(dict(end_loss=end_loss))
            # flow_mini_loss = torch.abs(output['flow']).sum()/128
            # total_losses.update(dict(flow_mini_loss=flow_mini_loss))

        if self.cfg.LOSS.pose_loss:
            pose_loss, others = self.poseloss(output)
            total_losses.update(pose_loss)

        for k, v in total_losses.items():
            total_loss += v

        total_losses.update(dict(loss=total_loss))

        return total_losses, others

