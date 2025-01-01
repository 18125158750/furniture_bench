# coding=gbk
from typing import Dict

import torch

import furniture_bench.controllers.control_utils as C

import torch

from ipdb import set_trace as bp


def diffik_factory(real_robot=True, *args, **kwargs):
    """
    ����΢�����˶�ѧ (DiffIK) ��������factory������

    ������
        real_robot (bool): �Ƿ�Ϊ��ʵ�����˴�����������Ĭ��Ϊ True��

    ���أ�
        DiffIKController ��
    """
    if real_robot:
        import torchcontrol as toco

        base = toco.PolicyModule
    else:
        base = torch.nn.Module

    class DiffIKController(base):
        """Differential Inverse Kinematics Controller"""
        """΢�����˶�ѧ������"""

        def __init__(
            self,
            pos_scalar=1.0,
            rot_scalar=1.0,
        ):
            """
            ��ʼ��΢�����˶�ѧ��������

            ������
                pos_scalar (float): λ�������������ӡ�Ĭ��Ϊ 1.0��
                rot_scalar (float): ��ת�����������ӡ�Ĭ��Ϊ 1.0��
            """
            super().__init__()
            self.ee_pos_desired = None
            self.ee_quat_desired = None
            self.ee_pos_error = None
            self.ee_rot_error = None

            self.pos_scalar = pos_scalar
            self.rot_scalar = rot_scalar

            self.joint_pos_desired = None

            self.scale_errors = True

            print(
                f"Making DiffIK controller with pos_scalar: {pos_scalar}, rot_scalar: {rot_scalar}"
            )

        def forward(
            self, state_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            """
            ǰ�򴫵ݺ��������㲢���������Ĺؽ�λ�á�

            ������
                state_dict (Dict[str, torch.Tensor]): ������ǰ״̬��Ϣ���ֵ䣬����ؽ�λ�á��ſɱȾ���ĩ��ִ����λ�˵ȡ�

            ���أ�
                Dict[str, torch.Tensor]: ��������õ��������ؽ�λ�õ��ֵ䡣
            """
            
            # ��ȡ״̬��Ϣ
            # joint_pos_current ����״: (batch_size, num_joints = 7)
            joint_pos_current = state_dict["joint_positions"]  # ��ǰ�ؽ�λ��
            if len(joint_pos_current.shape) == 1:
                joint_pos_current = joint_pos_current.unsqueeze(0)

            if self.joint_pos_desired is None:
                # ʹ�ó�ʼ�ؽ�λ����Ϊ���λ��
                self.joint_pos_desired = joint_pos_current
                # ��ʼ��kp����
                self.joint_pos_kp = torch.diag(torch.Tensor([50,50,50,50,50,50,50])).to(joint_pos_current.device)

            # jacobian ����״: (batch_size, 6, num_joints = 7)
            jacobian = state_dict["jacobian_diffik"]
            if len(jacobian.shape) == 2:
                jacobian = jacobian.unsqueeze(0)

            # ee_pos ����״: (batch_size, 3)
            # ee_quat ����״: (batch_size, 4)��ʵ����ĩβ
            ee_pos, ee_quat_xyzw = state_dict["ee_pos"], state_dict["ee_quat"]  # ĩ��ִ����λ�ú���̬����Ԫ����
            if len(ee_pos.shape) == 1:
                ee_pos = ee_pos.unsqueeze(0)
            if len(ee_quat_xyzw.shape) == 1:
                ee_quat_xyzw = ee_quat_xyzw.unsqueeze(0)
            goal_ori_xyzw = self.goal_ori  # Ŀ����̬����Ԫ����

            position_error = self.goal_pos - ee_pos

            # ����Ԫ��ת��Ϊ��ת����
            ee_mat = C.quaternion_to_matrix(ee_quat_xyzw)  # ��ǰĩ��ִ������̬����ת����
            goal_mat = C.quaternion_to_matrix(goal_ori_xyzw)  # Ŀ����̬����ת����

            # ����������
            mat_error = torch.matmul(goal_mat, torch.inverse(ee_mat))

            # ���������ת��Ϊ��Ǳ�ʾ
            ee_delta_axis_angle = C.matrix_to_axis_angle(mat_error)

            dt = 0.1  # ��������

            # ����������ĩ��ִ�������ٶȺͽ��ٶ�
            ee_pos_vel = position_error * self.pos_scalar / dt  # ���������ٶ�
            ee_rot_vel = ee_delta_axis_angle * self.rot_scalar / dt  # �����Ľ��ٶ�

            # �����������ٶȺͽ��ٶȺϲ�Ϊ�������ٶ�����
            ee_velocity_desired = torch.cat((ee_pos_vel, ee_rot_vel), dim=-1).unsqueeze(-1)

            # ���ݵ�ǰ�ؽڽǶȺͱ�ƹؽڽǶȼ���Ŀ��ؽ��ٶ�
            joint_vel_desired_by_pos = ((self.joint_pos_desired-joint_pos_current)@self.joint_pos_kp).unsqueeze(-1)

            # ���ſɱȾ����������ֵ�ֽ�
            U, S, V = torch.svd(jacobian,some=False)
            # ����ռ䣨�ݲ���������㣩
            # rtol = 1e-6
            # rank = (S > rtol * S[0]).sum(dim=1)
            rank = 6
            null_space = V[:,:,rank:]
            N = null_space.mT

            # ����qp�Ż�����
            lambda_ = 0.01
            A = jacobian
            P = 2 * (torch.bmm(A.mT, A) + lambda_ * torch.bmm(N.mT, N))
            Q = -2 * torch.bmm(A.mT, ee_velocity_desired)
            - 2 * lambda_ * torch.bmm(torch.bmm(N.mT, N),joint_vel_desired_by_pos)

            # ʹ���ſɱȾ����α����������Ĺؽ��ٶ�
            joint_vel_desired = torch.linalg.lstsq(
                P, -Q
            ).solution.squeeze(2)
            # joint_vel_desired = torch.linalg.lstsq(
            #     jacobian, ee_velocity_desired
            # ).solution.squeeze(2)

            # ���������Ĺؽ��ٶȼ��������Ĺؽ�λ��
            joint_pos_desired = joint_pos_current + joint_vel_desired * dt

            return {"joint_positions": joint_pos_desired}

        def set_goal(self, goal_pos, goal_ori):
            """
            ����Ŀ��λ�ú���̬��

            ������
                goal_pos (torch.Tensor): Ŀ��λ�á�
                goal_ori (torch.Tensor): Ŀ����̬����Ԫ������
            """
            self.goal_pos = goal_pos
            self.goal_ori = goal_ori

        def reset(self):
            """���ÿ�����"""
            pass

    return DiffIKController(*args, **kwargs)
