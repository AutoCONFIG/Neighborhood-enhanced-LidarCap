import torch.nn as nn
from util.geometry import axis_angle_to_rotation_matrix
from util.smpl import SMPL
from loss.zmp_loss import Zmp_Loss
from loss.bone_loss import Bone_Loss


class Pose_Loss(nn.Module):
    def __init__(self, ZMP, BONE):
        super().__init__()
        self.criterion_param = nn.MSELoss()
        self.criterion_joints = nn.MSELoss()
        self.criterion_vertices = nn.MSELoss()
        # self.chamfer_loss = ChamferLoss()
        self.smpl = SMPL().cuda()
        self.Zmp_loss = Zmp_Loss()
        self.Bone_loss = Bone_Loss()
        self.ZMP = ZMP
        self.BONE = BONE

    def forward(self, kw):
        B, T = kw['human_points'].shape[:2]
        gt_pose = kw['pose']
        gt_rotmats = axis_angle_to_rotation_matrix(
            gt_pose.reshape(-1, 3)).reshape(B, T, 24, 3, 3)

        #@mqh
        #gt_full_joints = kw['full_joints'].reshape(B, T, 24, 3)
        gt_full_joints = self.smpl.get_full_joints(self.smpl(gt_rotmats.reshape(-1, 24, 3, 3), gt_rotmats.new_zeros((B * T, 10)))).reshape(B, T, 24, 3) if 'full_joints' not in kw else kw['full_joints']
        gt_full_joints = gt_full_joints.detach()

        details = {}
        others = {}

        if 'pred_rotmats' in kw:
            pred_rotmats = kw['pred_rotmats'].reshape(B, T, 24, 3, 3)
            loss_param = self.criterion_param(pred_rotmats, gt_rotmats)
            details['loss_param'] = loss_param

            pred_human_vertices = self.smpl(
                pred_rotmats.reshape(-1, 24, 3, 3), pred_rotmats.new_zeros((B * T, 10)))
            pred_smpl_joints = self.smpl.get_full_joints(
                pred_human_vertices).reshape(B, T, 24, 3)
            loss_smpl_joints = self.criterion_joints(
                pred_smpl_joints, gt_full_joints)
            details['loss_smpl_joints'] = loss_smpl_joints
            others['pred_smpl_joints'] = pred_smpl_joints

            if self.ZMP:
                zmp_smpl_loss = self.Zmp_loss(kw, pred_smpl_joints)
                details['loss_zmp_smpl_joints'] = zmp_smpl_loss
                zmp_full_loss = self.Zmp_loss(kw, kw['pred_full_joints'])
                details['loss_zmp_full_joints'] = zmp_full_loss
            if self.BONE:
                bone_loss= self.Bone_loss(kw['pred_full_joints'])
                details['loss_bone_first'] = bone_loss
            # gt_human_vertices = self.smpl(
            #     gt_rotmats.reshape(-1, 24, 3, 3), torch.zeros((B * T, 10)).cuda())
            # loss_vertices = self.criterion_vertices(
            #     pred_human_vertices, gt_human_vertices)
            # details['loss_vertices'] = loss_vertices

        if 'pred_full_joints' in kw:
            pred_full_joints = kw['pred_full_joints']
            loss_full_joints = self.criterion_joints(
                pred_full_joints, gt_full_joints)
            details['loss_full_joints'] = loss_full_joints
        # human_points = kw['human_points'].reshape(BT, -1, 3)
        # loss_shape = chamfer_distance(human_points, kw['pred_vertices'])[0]
        # loss_shape.requires_grad_()

        # gt_human_vertex = kw['human_vertex']

        # loss = 0
        # for _, v in details.items():
        #     loss += v
        # details['loss'] = loss
        return details, others