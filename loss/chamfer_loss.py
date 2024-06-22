# -*- coding: utf-8 -*-
# @Author  : jingyi

import torch
import torch.nn as nn

class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        '''
        :param x: (bs, np, 3)
        :param y: (bs, np, 3)
        :return: loss
        '''
        B, T, N,_ = x.size()
        x = x.view(B*T,N,-1).contiguous()
        y = y.view(B*T,N,-1).contiguous()
        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        dist = torch.sqrt(1e-6 +
                          torch.sum(torch.pow(x - y, 2), 3))  # bs, ny, nx
        min1, _ = torch.min(dist, 1)
        min2, _ = torch.min(dist, 2)

        return min1.mean() + min2.mean()