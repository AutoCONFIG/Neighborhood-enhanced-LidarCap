import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

from util.geometry import rot6d_to_rotmat
from model.st_gcn import STGCN
from model.pointnet2 import PointNet2Encoder
from model.transformer import TransformerEncoder
from math import sqrt
from model.diffusion_trans import MDM_Trans

vgg16 = models.vgg16(pretrained=True)
vgg16.classifier._modules['3'] = nn.Linear(4096,2048)
vgg16.classifier._modules['6'] = nn.Linear(2048,1024)


class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(n_hidden, n_hidden, n_rnn_layer,
                          batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(n_input, n_hidden)

        self.linear2 = nn.Linear(n_hidden * 2, n_output)

    def forward(self, x):  # (B, T, D)
        x = self.rnn(F.relu(F.dropout(self.linear1(x)), inplace=True))[0]       # 此处F.dropout用法有误，但估计不影响性能
        return self.linear2(x)


class Outline(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        out_dim=1024
        if cfg.MODEL.use_back_wang:
            self.encoder2 = PointNet2Encoder(outdim=256)
            self.encoder_back = PointNet2Encoder(outdim=256)
            out_dim=512

        if cfg.MODEL.use_attention:
            self.encoder2 = PointNet2Encoder(outdim=256)
            self.encoder_back = PointNet2Encoder(outdim=256)
            out_dim=512
            self.attention_linear1 = nn.Linear(1024,256)
            self.attention_linear2 = nn.Linear(256,3)

        if cfg.MODEL.use_transformer:
            self.outline_encoder = PointNet2Encoder(outdim=1024)
            self.background_encoder = PointNet2Encoder(outdim=1024)
            out_dim=1024
            
        if cfg.MODEL.use_mdmtransformer:
            self.outline_encoder = PointNet2Encoder(outdim=512)
            # self.background_encoder = PointNet2Encoder(outdim=512)
            self.background_encoder = PointNet2Encoder(outdim=512)
            out_dim=512

            self.fc_layer = nn.Sequential(
                nn.Linear(1536, 1024, bias=False),
                nn.BatchNorm1d(16),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(1024, 1024),
            )

        if cfg.MODEL.Outline_boundary:
            self.encoder2 = PointNet2Encoder(outdim=512)
            out_dim=512
            
        if cfg.MODEL.add_background:
            self.encoder2 = PointNet2Encoder(outdim=512)
            out_dim=512
            
        if cfg.MODEL.add_twise_noise:
            self.encoder2 = PointNet2Encoder(outdim=512)
            out_dim=512

        self.encoder = PointNet2Encoder(outdim=out_dim)
            
        self.pose_s1 = RNN(1024, 24 * 3, 1024)

        self.pose_s2 = STGCN(3 + 1024)
        
        self.cfg = cfg

        self.vgg16 = vgg16

        self.transformer = TransformerEncoder(cfg)
        
        self.mdmtransformer = MDM_Trans(cfg)

    def forward(self, data):

        pred = {}
        B, T, N, _ = data['human_points'].shape

        if self.cfg.MODEL.use_project_image:
            i_B, i_T, i_C, i_H, i_W = data['project_image'].shape
            x = self.vgg16(data['project_image'].reshape(-1,i_C, i_H, i_W))
            x = x.reshape(i_B, i_T, -1)
        else:
            x = self.encoder(data['human_points'])  # (B, T, D)

            if self.cfg.MODEL.Outline_boundary:
                x1 = self.encoder2(data['human_boundary'])
                x = torch.cat([x,x1],-1)
            
            if self.cfg.MODEL.add_background:
                x1 = self.encoder2(data['back_pc'])
                x = torch.cat([x,x1],-1)
            
            if self.cfg.MODEL.add_twise_noise:
                x1 = self.encoder2(data['twice_noise'])
                x = torch.cat([x,x1],-1)
                
            if self.cfg.MODEL.use_back_wang:
                x1 = self.encoder2(data['twice_noise'])
                x2 = self.encoder_back(data['back_pc'])
                x_tmp = torch.cat([x1,x2], dim=-1)
                x = torch.cat([x,x_tmp], dim=-1)

            if self.cfg.MODEL.use_attention:
                x1 = self.encoder2(data['human_boundary'])
                x2 = self.encoder_back(data['back_pc'])
                x_tmp = torch.cat([x1,x2], dim=-1)
                x_all = torch.cat([x,x_tmp], dim=-1)

                x_all = x_all.reshape(B*T,-1)
                weight_attention = F.softmax(self.attention_linear2(
                    F.relu(F.dropout(self.attention_linear1(x_all)))),dim=-1)
                weight_attention = weight_attention.reshape(B,T,-1)

                x_addi = torch.cat([x1*weight_attention[...,0:1],
                                    x2*weight_attention[...,1:2]], dim=-1)
                x = torch.cat([x*weight_attention[...,2:3], x_addi], dim=-1)

            if self.cfg.MODEL.use_transformer:
                x_human = x
                x_background = self.background_encoder(data['back_pc'])
                x_boundary = self.outline_encoder(data['human_boundary'])
                x_embedding = x_human + x_boundary + x_background
                x = self.transformer(x_embedding)
                
            if self.cfg.MODEL.use_mdmtransformer:
                x_human = x
                x_background = self.background_encoder(data['back_pc'])
                x_boundary = self.outline_encoder(data['twice_noise'])
                x_tmp = torch.cat([x_background,x_boundary], dim=-1)
                x = torch.cat([x_human,x_tmp], dim=-1)
                # x = torch.cat([x, x_background],-1)
                x_gloabal = self.fc_layer(x)
                
        # x: (B,T,D)
        if self.cfg.MODEL.use_mdmtransformer:
            full_joints= self.mdmtransformer(x_human, x_boundary,
                                                              x_background,
                                                              self.cfg.TrainDataset.seqlen,
                                                              data['trans'])
            rot6ds = self.pose_s2(torch.cat((full_joints.reshape(
                B, T, 24, 3), x_gloabal.unsqueeze(-2).repeat(1, 1, 24, 1)), dim=-1))
        else:
            full_joints = self.pose_s1(x)  # (B, T, 24, 3)
            rot6ds = self.pose_s2(torch.cat((full_joints.reshape(
                B, T, 24, 3), x.unsqueeze(-2).repeat(1, 1, 24, 1)), dim=-1))

        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))  # (B * T, D)
        rotmats = rot6d_to_rotmat(
            rot6ds).reshape(-1, 3, 3)  # (B * T * 24, 3, 3)
        pred['pred_rotmats'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['pred_full_joints'] = full_joints.reshape(B, T, 24, 3)
        pred = {**data, **pred}
        return pred
