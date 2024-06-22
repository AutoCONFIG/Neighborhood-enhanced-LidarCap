# -*- coding: utf-8 -*-
# @Author  : jingyi
import torch
import torch.nn as nn

def unsqueeze_gen_data(generated_data, dev='cuda'):
    """
    Add batch dimension for generated_data
    For now only supports batch size = 1

    :param generated_data:
    """
    for data_map in generated_data:
        for k in data_map:
            if not torch.is_tensor(data_map[k]):
                data_map[k] = torch.tensor(data_map[k])
            data_map[k] = torch.unsqueeze(data_map[k], 0).to(dev)

def get_gen_data(pc1, pc2, data_gen, max_layers=None):
    # compute generated data needed for the model
    data = data_gen.compute_generated_data(torch.squeeze(pc1).cpu(), torch.squeeze(pc2).cpu(), max_layers)
    # add batch dimension
    unsqueeze_gen_data(data)
    return data

def compute_pairwise_dist(a, b):
    """
    Compute similarity matrix
    D_i,j is distance between a[:, :, i] and b[:, :, j]

    :param a:
    :param b:
    :return: dist
    """
    with torch.no_grad():
        r_a = torch.sum(a * a, dim=1, keepdim=True)  # (B,1,N)
        r_b = torch.sum(b * b, dim=1, keepdim=True)  # (B,1,M)
        mul = torch.matmul(a.permute(0, 2, 1), b)    # (B,N,M)
        dist = r_a.permute(0, 2, 1) - 2 * mul + r_b  # (B,N,M)
    return dist


def batched_index_select(x, dim, index):
    """
    Analog of index_select across batches

    :param x:
    :param dim:
    :param index:
    :return: Selected values across batches
    """
    for ii in range(1, len(x.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(x.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(x, dim, index)

def get_nn(pc1, pc2):
    """
    # Finds NN for each point in pc1 from pc2

    Args:
        pc1: human points (ori)
        pc2: sample points (tar)

    Returns: nn

    """
    dist = compute_pairwise_dist(pc1, pc2)
    nn_idx = torch.argmin(dist, dim=2)
    nn_pc1 = batched_index_select(pc2, 2, nn_idx)
    return nn_pc1


class NN_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        '''
        :param pred: (bs, np, 3)
        :param target: (bs, np, 3)
        :return: loss
        '''
        B, T, N,_ = pred.size()
        pred = pred.view(B*T,N,-1).contiguous()
        target = target.view(B*T,N,-1).contiguous()
        
        # find NN for each point in pred from target
        nn_target = get_nn(pred, target)
        
        return torch.norm(pred - nn_target, p=2, dim=1)