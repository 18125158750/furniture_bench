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

            # jacobian ����״: (batch_size, 6, num_joints = 7)
            jacobian = state_dict["jacobian_diffik"]

            # ee_pos ����״: (batch_size, 3)
            # ee_quat ����״: (batch_size, 4)��ʵ����ĩβ
            ee_pos, ee_quat_xyzw = state_dict["ee_pos"], state_dict["ee_quat"]  # ĩ��ִ����λ�ú���̬����Ԫ����
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
            ee_velocity_desired = torch.cat((ee_pos_vel, ee_rot_vel), dim=-1)

            # ʹ���ſɱȾ����α����������Ĺؽ��ٶȣ�Ч�ʺܵͣ�
            # joint_vel_desired = torch.linalg.lstsq(
            #     jacobian, ee_velocity_desired
            # ).solution
            
            pinv = torch.linalg.pinv(jacobian)
            joint_vel_desired = torch.matmul(pinv, ee_velocity_desired.unsqueeze(-1)).squeeze(-1)

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
