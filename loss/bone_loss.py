# -*- coding: utf-8 -*-
# @Author  : jingyi

import torch
import torch.nn as nn


class Bone_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_joints):
        # pred_joints: (B, T, 24, 3)
        #leg
        left_hip_joints = pred_joints[:, :, 1, :]
        left_knee_joints = pred_joints[:, :, 4, :]
        left_ankle_joints = pred_joints[:, :, 7, :]
        left_toe_joints = pred_joints[:, :, 10, :]

        right_hip_joints = pred_joints[:, :, 2, :]
        right_knee_joints = pred_joints[:, :, 5, :]
        right_ankle_joints = pred_joints[:, :, 8, :]
        right_toe_joints = pred_joints[:, :, 11, :]

        #arm
        left_shoulder_joints = pred_joints[:, :, 16, :]
        left_elbow_joints = pred_joints[:, :, 18, :]
        left_wrist_joints = pred_joints[:, :, 20, :]
        left_finger_joints = pred_joints[:, :, 22, :]

        right_shoulder_joints = pred_joints[:, :, 17, :]
        right_elbow_joints = pred_joints[:, :, 19, :]
        right_wrist_joints = pred_joints[:, :, 21, :]
        right_finger_joints = pred_joints[:, :, 23, :]

        # mid
        neck_joints = pred_joints[:, :, 12, :]
        waist_joints = pred_joints[:, :, 0, :]

        #dis leg
        dis_left_upper_leg =  torch.norm(left_hip_joints-left_knee_joints,dim=2)
        dis_left_low_leg =  torch.norm(left_ankle_joints-left_knee_joints,dim=2)
        dis_left_foot =  torch.norm(left_ankle_joints-left_toe_joints,dim=2)
        dis_left_hip =  torch.norm(waist_joints-left_hip_joints,dim=2)

        dis_right_upper_leg =  torch.norm(right_hip_joints-right_knee_joints,dim=2)
        dis_right_low_leg =  torch.norm(right_ankle_joints-right_knee_joints,dim=2)
        dis_right_foot =  torch.norm(right_ankle_joints-right_toe_joints,dim=2)
        dis_right_hip =  torch.norm(waist_joints-right_hip_joints,dim=2)

        dis_leg = torch.abs(dis_left_upper_leg-dis_right_upper_leg)+\
                  torch.abs(dis_left_low_leg-dis_right_low_leg)+\
                  torch.abs(dis_left_foot-dis_right_foot)+\
                  torch.abs(dis_left_hip-dis_right_hip)

        #dis arm
        dis_right_upper_arm = torch.norm(right_shoulder_joints-right_elbow_joints,dim=2)
        dis_right_low_arm = torch.norm(right_wrist_joints-right_elbow_joints,dim=2)
        dis_right_hand = torch.norm(right_wrist_joints-right_finger_joints,dim=2)
        dis_right_shoulder = torch.norm(right_shoulder_joints-neck_joints,dim=2)

        dis_left_upper_arm = torch.norm(left_shoulder_joints-left_elbow_joints,dim=2)
        dis_left_low_arm = torch.norm(left_wrist_joints-left_elbow_joints,dim=2)
        dis_left_hand = torch.norm(left_wrist_joints-left_finger_joints,dim=2)
        dis_left_shoulder = torch.norm(left_shoulder_joints-neck_joints,dim=2)

        dis_arm = torch.abs(dis_left_upper_arm-dis_right_upper_arm)+\
                  torch.abs(dis_left_low_arm-dis_right_low_arm)+\
                  torch.abs(dis_left_hand-dis_right_hand)+\
                  torch.abs(dis_left_shoulder-dis_right_shoulder)

        dis_all = dis_arm/4 + dis_leg/4
        return dis_all.mean()


if __name__ == '__main__':
    pass
