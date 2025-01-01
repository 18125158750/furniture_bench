# coding=gbk
"""Code derived from https://github.com/StanfordVL/perls2 and https://github.com/ARISE-Initiative/robomimic"""
import math
from typing import Dict, List

import torch
import sys
import furniture_bench.controllers.control_utils as C


def osc_factory(real_robot=True, *args, **kwargs):
    """
    ���ڴ��������ռ���� (OSC) �������Ĺ���������

    Args:
        real_robot (bool): �������������ʵ�Ļ�����һ��ʹ�ã���Ϊ True������Ϊ False��
                           ������˿������Ļ��� (������ʵ������Ϊ torchcontrol.PolicyModule��
                           ����Ϊ torch.nn.Module)��
        *args: �ɱ䳤�Ȳ����б�
        **kwargs: ����ؼ��ֲ�����

    Returns:
        OSCController: OSCController ���һ��ʵ����
    """
    if real_robot:
        import torchcontrol as toco

        base = toco.PolicyModule
    else:
        base = torch.nn.Module

    class OSCController(base):
        """
        ���ڿ��ƻ�е��������ռ����˶��Ĳ����ռ���� (OSC) �ࡣ
        """

        def __init__(
            self,
            kp: torch.Tensor,
            kv: torch.Tensor,
            ee_pos_current: torch.Tensor,
            ee_quat_current: torch.Tensor,
            init_joints: torch.Tensor,
            position_limits: torch.Tensor,
            mass_matrix_offset_val: List[float] = [0.2, 0.2, 0.2],
            max_dx: float = 0.005,
            controller_freq: int = 1000,
            policy_freq: int = 5,
            ramp_ratio: float = 1,
            joint_kp: float = 10.0,
        ):
            """
            ��ʼ��ĩ��ִ�����迹��������

            Args:
                kp (torch.Tensor): ���ڸ���λ��/�������ȷ������Ť�ص�λ�����档
                                    �����Ǳ��������ж���ά�ȵ�ֵ��ͬ�����б�ÿ��ά�ȶ����ض���ֵ����
                kv (torch.Tensor): ���ڸ����ٶ�/���ٶ����ȷ������Ť�ص��ٶ����档
                                    �����Ǳ��������ж���ά�ȵ�ֵ��ͬ�����б�ÿ��ά�ȶ����ض���ֵ����
                                    ��������� kv,��������ᡣ
                ee_pos_current (torch.Tensor): ��ǰĩ��ִ������λ�á�
                ee_quat_current (torch.Tensor): ��ǰĩ��ִ�����ķ���
                init_joints (torch.Tensor): ��ʼ�ؽ�λ�ã�������ռ䣩��
                position_limits (torch.Tensor): �������Ŀ��ĩ��ִ����λ�ô�С������������Щ���ƣ��ף�֮�ں�֮�ϡ�
                                                ������ 2 Ԫ���б����еѿ���ά�ȵ���С/���ֵ��ͬ��
                                                �� 2 Ԫ���б���б�ÿ��ά�ȶ����ض�����С/���ֵ����
                mass_matrix_offset_val (list): Ҫ��ӵ���������Խ����������Ԫ�ص�ƫ������ 3f �б�
                                                ������ʵ�����ˣ��Ե���ĩ�˹ؽڴ��ĸ�Ħ������
                max_dx (float): ��ֵ������λ���ƶ������������
                control_freq (int): ����ѭ����Ƶ�ʡ�
                policy_freq (int): �ӻ����˲�����˿��������Ͷ�����Ƶ�ʡ�
                ramp_ratio (float): control_freq / policy_freq �ı��ʡ�����ȷ����ֵ����Ҫ��ȡ�Ĳ�����
                joint_kp (float): �ؽ�λ�ÿ��Ƶı������档
            """
            super().__init__()
            # limits
            # ĩ��ִ����λ�õ�����
            self.position_limits = position_limits
            # ����������
            self.kp = kp
            self.kv = kv
            # ��ʼ�ؽ�λ��
            self.init_joints = init_joints

            # ������ĩ��ִ����λ�úͷ��򣨿����Ż��Ĳ�����
            self.ee_pos_desired = torch.nn.Parameter(ee_pos_current)
            self.ee_quat_desired = torch.nn.Parameter(ee_quat_current)

            # ��ʵ�����˵���������ƫ��ֵ
            # self.mass_matrix = torch.zeros((7, 7))
            self.mass_matrix_offset_val = mass_matrix_offset_val
            self.mass_matrix_offset_idx = torch.tensor([[4, 4], [5, 5], [6, 6]])

            # ���ڸ����ظ�Ť�صı���
            self.repeated_torques_counter = 0
            self.num_repeated_torques = 3
            self.prev_torques = torch.zeros((7,))

            # λ�ò�ֵ������
            # Interpolator pos, ori
            self.max_dx = max_dx  # Maximum allowed change per interpolator step
            self.total_steps = math.floor(
                ramp_ratio * float(controller_freq) / float(policy_freq)
            ) # ÿ����ֵ���������ܲ���

            # ���ڲ�ֵ�ĵ�ǰĿ��λ�ú���ǰĿ��λ��
            self.goal_pos = ee_pos_current.clone()
            self.prev_goal_pos = ee_pos_current.clone()
            self.step_num_pos = 1

            # �����ֵ������
            self.fraction = 0.5
            self.goal_ori = ee_quat_current.clone()
            self.prev_goal_ori = ee_quat_current.clone()
            self.step_num_ori = 1

            # ��ǰ��ֵ��λ�úͷ���
            self.prev_interp_pos = ee_pos_current.clone()
            self.prev_interp_ori = ee_quat_current.clone()

            # �ؽڱ�������
            self.joint_kp = joint_kp

        def forward(
            self, state_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            """
            �����������򴫵ݣ����ݵ�ǰ״̬��������Ĺؽ�Ť�ء�

            Args:
                state_dict (Dict[str, torch.Tensor]): ������ǰ������״̬���ֵ䡣 
                                                        Ӧ�������¼���'joint_positions'��'joint_velocities'�� 
                                                        'mass_matrix'��'ee_pose'��'jacobian'��

            Returns:
                Dict[str, torch.Tensor]: ����������Ĺؽ�Ť�أ�'joint_torques'�����ֵ䡣
            """

            # �����ظ�Ť�ؼ����������ڱ�Ҫʱ������ǰ��Ť��
            self.repeated_torques_counter = (
                self.repeated_torques_counter + 1
            ) % self.num_repeated_torques
            if self.repeated_torques_counter != 1:
                return {"joint_torques": self.prev_torques}
            
            # ��ȡ��ǰ�ؽ�λ�ú��ٶ�
            joint_pos_current = state_dict["joint_positions"] # ʼ��Ϊtorch.Size([7])
            joint_vel_current = state_dict["joint_velocities"]

            # ��ȡ��������Ϊ��ʵ���������ƫ����
            mass_matrix = state_dict["mass_matrix"].reshape(7, 7).t()
            mass_matrix[4, 4] += self.mass_matrix_offset_val[0]
            mass_matrix[5, 5] += self.mass_matrix_offset_val[1]
            mass_matrix[6, 6] += self.mass_matrix_offset_val[2]

            # ��ȡĩ��ִ����λ�˲���ȡλ�úͷ���
            ee_pose = state_dict["ee_pose"].reshape(4, 4).t().contiguous()

            ee_pos, ee_quat = C.mat2pose(ee_pose)
            ee_pos = ee_pos.to(ee_pose.device)
            ee_quat = ee_quat.to(ee_pose.device)

            # ��ȡ�ſɱȾ���
            jacobian = state_dict["jacobian"].reshape(7, 6).t().contiguous()

            # ���㵱ǰĩ��ִ�������ٶȣ����ٶȺͽ��ٶȣ�
            ee_twist_current = jacobian @ joint_vel_current
            ee_pos_vel = ee_twist_current[:3]
            ee_ori_vel = ee_twist_current[3:]

            # ����Ŀ��λ�úͷ��򣬲��ڱ�Ҫʱ��λ�ý��вü�
            goal_pos = C.set_goal_position(self.position_limits, self.ee_pos_desired)
            goal_ori = self.ee_quat_desired

            # �������ڲ�ֵ�ĵ�ǰĿ��λ�úͷ���
            self.set_goal(goal_pos, goal_ori)

            # ��ȡ��ֵ���Ŀ��λ�úͷ���
            goal_pos = self.get_interpolated_goal_pos()
            goal_ori = self.get_interpolated_goal_ori()

            # ��Ŀ�귽�����Ԫ��ת��Ϊ��ת����
            goal_ori_mat = C.quat2mat(goal_ori).to(goal_ori.device)
            ee_ori_mat = C.quat2mat(ee_quat).to(ee_quat.device)

            # ���㷽�����
            ori_error = C.orientation_error(goal_ori_mat, ee_ori_mat)

            # ʹ�ÿ����ɼ���ĩ��ִ����������������Ť��
            position_error = goal_pos - ee_pos
            vel_pos_error = -ee_pos_vel
            desired_force = torch.multiply(
                position_error, self.kp[0:3]
            ) + torch.multiply(vel_pos_error, self.kv[0:3])

            vel_ori_error = -ee_ori_vel
            desired_torque = torch.multiply(ori_error, self.kp[3:]) + torch.multiply(
                vel_ori_error, self.kv[3:]
            )

            # ��������ռ������������ռ����
            lambda_full, nullspace_matrix = C.opspace_matrices(mass_matrix, jacobian)

            # ������������/���أ����֣�
            desired_wrench = torch.cat([desired_force, desired_torque])

            # �����ֽ���Ϊ����ռ����ռ����
            decoupled_wrench = torch.matmul(lambda_full, desired_wrench)

            # �������İ���ͶӰ���ؽ�Ť����
            torques = torch.matmul(jacobian.T, decoupled_wrench) + C.nullspace_torques(
                mass_matrix,
                nullspace_matrix,
                self.init_joints,
                joint_pos_current,
                joint_vel_current,
                joint_kp=self.joint_kp,
            )

            # Ӧ��Ť��ƫ���Է�ֹ�����˿�ס
            self._torque_offset(ee_pos, goal_pos, torques)

            # ����������Ť�ز�����
            self.prev_torques = torques

            return {"joint_torques": torques}

        def set_goal(self, goal_pos, goal_ori):
            """
            ���ÿ�������Ŀ��λ�úͷ���

            �˷�������Ŀ��λ�úͷ��򣬲�����������Ŀ�����ǰ�Ĳ�ֵ���ʱ���ò�ֵ��

            Args:
                goal_pos (torch.Tensor): ĩ��ִ����������Ŀ��λ�á�
                goal_ori (torch.Tensor): ĩ��ִ����������Ŀ�귽��
            """
            if (
                not torch.isclose(goal_pos, self.goal_pos).all()
                or not torch.isclose(goal_ori, self.goal_ori).all()
            ):
                self.prev_goal_pos = self.goal_pos.clone()
                self.goal_pos = goal_pos.clone()
                self.step_num_pos = 1

                self.prev_goal_ori = self.goal_ori.clone()
                self.goal_ori = goal_ori.clone()
                self.step_num_ori = 1
            elif (
                self.step_num_pos >= self.total_steps
                or self.step_num_ori >= self.total_steps
            ):
                self.prev_goal_pos = self.prev_interp_pos.clone()
                self.goal_pos = goal_pos.clone()
                self.step_num_pos = 1

                self.prev_goal_ori = self.prev_interp_ori.clone()
                self.goal_ori = goal_ori.clone()
                self.step_num_ori = 1

        def get_interpolated_goal_pos(self) -> torch.Tensor:
            """
            ���㲢���ز�ֵ���Ŀ��λ�á�

            �˷������ݵ�ǰλ�á�Ŀ��λ�úͲ�ֵ����������һ����ֵĿ��λ�á�

            Returns:
                torch.Tensor: ��ֵ���Ŀ��λ�á�
            """
            # Calculate the desired next step based on remaining interpolation steps and increment step if necessary
            dx = (self.goal_pos - self.prev_goal_pos) / (self.total_steps)
            # Check if dx is greater than max value; if it is; clamp and notify user
            if torch.any(abs(dx) > self.max_dx):
                dx = torch.clip(dx, -self.max_dx, self.max_dx)

            interp_goal = self.prev_goal_pos + (self.step_num_pos + 1) * dx
            self.step_num_pos += 1
            self.prev_interp_pos = interp_goal
            return interp_goal

        def get_interpolated_goal_ori(self):
            """
            ʹ���������Բ�ֵ (slerp) ���㲢���ز�ֵ���Ŀ�귽��

            Returns:
                torch.Tensor: ��ֵ���Ŀ�귽��
            """
            """Get interpolated orientation using slerp."""
            interp_fraction = (self.step_num_ori / self.total_steps) * self.fraction
            interp_goal = C.quat_slerp(
                self.prev_goal_ori, self.goal_ori, fraction=interp_fraction
            )
            self.step_num_ori += 1
            self.prev_interp_ori = interp_goal

            return interp_goal

        def _torque_offset(self, ee_pos, goal_pos, torques):
            """
            Ӧ��Ť��ƫ���Է�ֹ��������λ�����ƴ���ס��

            �˷������ĩ��ִ�����Ƿ�λ��λ�����ƴ���Զ��Ŀ���ƶ���
            ����ǣ���Ӧ��Ť��ƫ���԰�����������Ŀ���ƶ���

            Args:
                ee_pos (torch.Tensor): ��ǰĩ��ִ������λ�á�
                goal_pos (torch.Tensor): ������Ŀ��λ�á�
                torques (torch.Tensor): ������Ĺؽ�Ť�ء�
            """
            """Torque offset to prevent robot from getting stuck when reached too far."""
            if (
                ee_pos[0] >= self.position_limits[0][1]
                and goal_pos[0] - ee_pos[0] <= -self.max_dx
            ):
                torques[1] -= 2.0
                torques[3] -= 2.0

        def reset(self):
            self.repeated_torques_counter = 0

    return OSCController(*args, **kwargs)
