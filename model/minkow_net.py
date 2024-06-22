import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util.geometry import rot6d_to_rotmat
from model.st_gcn import STGCN
from model.pointnet2 import PointNet2Encoder
from model.minkow_pointnet import MinkowskiPointNet
from model.outline import RNN
import MinkowskiEngine as ME

class MinkowNet(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.encoder = MinkowskiPointNet()
        self.pose_s1 = RNN(1024, 24 * 3, 1024)

        self.pose_s2 = STGCN(3 + 1024)
        self.cfg = cfg
        
        if cfg.MODEL.use_background:
            self.back_encoder = MinkowskiPointNet()
            self.conv1 = torch.nn.Conv1d(2048, 1024,1)
            self.conv2 = torch.nn.Conv1d(1024, 1024,1)
            self.bn1 = nn.GroupNorm(16, 1024)
            
    def forward(self, data):
        pred = {}       
        B, T, N, _ = data['human_points'].shape
            
        minknet_input = ME.TensorField(
            coordinates=data['coordinates'], features=data['features']
        )
        x = self.encoder(minknet_input)  # (B, T, D)
        
        if self.cfg.MODEL.use_background:
            background_input = ME.TensorField(
                coordinates=data['background_coordinates'],
                features=data['background_features']
            )
            back_x = self.back_encoder(background_input)

            back_feat= back_x[0:1,:].repeat(T,1)
            for b in range(1,B):
                back_feat = torch.cat([back_feat, back_x[b:b+1,:].repeat(T,1)], dim=0)

            feat = torch.cat([x, back_feat], dim=1)
            feat = feat.reshape(B,T,-1).permute(0,2,1)
            # process to get latent features output
            feat = F.relu(self.bn1(self.conv1(feat)))
            x = self.conv2(feat).permute(0,2,1)
        x = x.reshape(B,T,-1)
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
