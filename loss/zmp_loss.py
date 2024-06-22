# -*- coding: utf-8 -*-
# @Author  : jingyi

import torch
import torch.nn as nn


class Zmp_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, kw, pred_joints):
        # pred_joints: (B, T, 24, 3)
        zmp = self.calculate_zmp(pred_joints)
        dis = self.point_to_plane_distance(zmp, kw['plane_model'])

        return dis.mean()

    def point_to_plane_distance(self, point, plane):
        # 计算点到平面的距离
        # 输入：
        #   point: (B, T, 3)的张量，代表B个序列中每个时刻T的点，每个点有三个坐标值
        #   plane: (B, T, 4)的张量，代表B个序列中每个时刻T的平面参数，前三个分别是平面法向量的x、y、z分量，第四个是平面常量d
        # 输出：
        #   distance: (B, T)的张量，代表B个序列中每个时刻T点到平面的距离

        # 将平面法向量的前三个分量提取出来，作为法向量
        plane_normal = plane[..., :3]

        # 计算点到平面的距离公式
        distance = torch.abs(
            torch.sum(plane_normal * point, dim=-1) + plane[..., 3]) / torch.norm(
            plane_normal, dim=-1)

        return distance

    def calculate_zmp(self, pred_joints):
        """
        计算ZMP
        Args:
            pred_joints (torch.Tensor): 预测的人体关节点位置，shape为（B，T，24，3）

        Returns:
            zmp (torch.Tensor): 计算得到的ZMP，shape为（B，T，3）
        """
        # 计算中心点
        com = pred_joints.mean(dim=2)  # 在24个关节点上取平均值，shape为（B，T，3）

        # 计算支撑面积
        left_hip_joints = pred_joints[:, :, 1:2, :]
        left_knee_joints = pred_joints[:, :, 4:5, :]
        left_ankle_joints = pred_joints[:, :, 7:8, :]
        left_toe_joints = pred_joints[:, :, 10:11, :]

        right_hip_joints = pred_joints[:, :, 2:3, :]
        right_knee_joints = pred_joints[:, :, 5:6, :]
        right_ankle_joints = pred_joints[:, :, 8:9, :]
        right_toe_joints = pred_joints[:, :, 11:12, :]

        foot_joints = torch.cat((left_hip_joints,left_knee_joints,left_ankle_joints,
                                 left_toe_joints,right_hip_joints,right_knee_joints,
                                 right_ankle_joints,right_toe_joints),
                                dim=2)  # 合并左右脚关节点，shape为（B，T，12，3）
        x_min, _ = foot_joints[:, :, :, 0].min(dim=2)  # 取最小值，shape为（B，T）
        x_max, _ = foot_joints[:, :, :, 0].max(dim=2)  # 取最大值，shape为（B，T）
        y_min, _ = foot_joints[:, :, :, 1].min(dim=2)  # 取最小值，shape为（B，T）
        y_max, _ = foot_joints[:, :, :, 1].max(dim=2)  # 取最大值，shape为（B，T）
        support_area = (x_max - x_min) * (y_max - y_min)  # 支撑面积，shape为（B，T）

        foot_joints_mean = foot_joints.mean(dim=2)
        # 计算ZMP
        zmp_x = com[:, :, 0] - (foot_joints_mean[:, :, 0] - com[:, :, 0]) * (
                    1 - torch.exp(-support_area / 10000)) / support_area
        zmp_y = com[:, :, 1] - (foot_joints_mean[:, :, 1] - com[:, :, 1]) * (
                    1 - torch.exp(-support_area / 10000)) / support_area
        zmp = torch.stack((zmp_x, zmp_y, torch.zeros_like(zmp_x)),
                          dim=2)  # shape为（B，T，3）

        return zmp


if __name__ == '__main__':
    pass
