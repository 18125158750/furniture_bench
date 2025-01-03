from typing import Dict

import torch

import furniture_bench.controllers.control_utils as C

import torch

from ipdb import set_trace as bp


def diffik_factory(real_robot=True, *args, **kwargs):
    """
    创建微分逆运动学 (DiffIK) 控制器的factory函数。

    参数：
        real_robot (bool): 是否为真实机器人创建控制器。默认为 True。

    返回：
        DiffIKController 类
    """
    if real_robot:
        import torchcontrol as toco

        base = toco.PolicyModule
    else:
        base = torch.nn.Module

    class DiffIKController(base):
        """Differential Inverse Kinematics Controller"""
        """微分逆运动学控制器"""

        def __init__(
            self,
            pos_scalar=1.0,
            rot_scalar=1.0,
        ):
            """
            初始化微分逆运动学控制器。

            参数：
                pos_scalar (float): 位置误差的缩放因子。默认为 1.0。
                rot_scalar (float): 旋转误差的缩放因子。默认为 1.0。
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
            前向传递函数，计算并返回期望的关节位置。

            参数：
                state_dict (Dict[str, torch.Tensor]): 包含当前状态信息的字典，例如关节位置、雅可比矩阵、末端执行器位姿等。

            返回：
                Dict[str, torch.Tensor]: 包含计算得到的期望关节位置的字典。
            """
            
            # 获取状态信息
            # joint_pos_current 的形状: (batch_size, num_joints = 7)
            joint_pos_current = state_dict["joint_positions"]  # 当前关节位置
            if len(joint_pos_current.shape) == 1:
                joint_pos_current = joint_pos_current.unsqueeze(0)

            if self.joint_pos_desired is None:
                # 使用初始关节位置作为标称位置
                self.joint_pos_desired = joint_pos_current
                # 初始化kp矩阵
                self.joint_pos_kp = torch.diag(torch.Tensor([50,50,50,50,50,50,50])).to(joint_pos_current.device)

            # jacobian 的形状: (batch_size, 6, num_joints = 7)
            jacobian = state_dict["jacobian_diffik"]
            if len(jacobian.shape) == 2:
                jacobian = jacobian.unsqueeze(0)

            # ee_pos 的形状: (batch_size, 3)
            # ee_quat 的形状: (batch_size, 4)，实部在末尾
            ee_pos, ee_quat_xyzw = state_dict["ee_pos"], state_dict["ee_quat"]  # 末端执行器位置和姿态（四元数）
            if len(ee_pos.shape) == 1:
                ee_pos = ee_pos.unsqueeze(0)
            if len(ee_quat_xyzw.shape) == 1:
                ee_quat_xyzw = ee_quat_xyzw.unsqueeze(0)
            goal_ori_xyzw = self.goal_ori  # 目标姿态（四元数）

            position_error = self.goal_pos - ee_pos

            # 将四元数转换为旋转矩阵
            ee_mat = C.quaternion_to_matrix(ee_quat_xyzw)  # 当前末端执行器姿态的旋转矩阵
            goal_mat = C.quaternion_to_matrix(goal_ori_xyzw)  # 目标姿态的旋转矩阵

            # 计算矩阵误差
            mat_error = torch.matmul(goal_mat, torch.inverse(ee_mat))

            # 将矩阵误差转换为轴角表示
            ee_delta_axis_angle = C.matrix_to_axis_angle(mat_error)

            dt = 0.1  # 控制周期

            # 计算期望的末端执行器线速度和角速度
            ee_pos_vel = position_error * self.pos_scalar / dt  # 期望的线速度
            ee_rot_vel = ee_delta_axis_angle * self.rot_scalar / dt  # 期望的角速度

            # 将期望的线速度和角速度合并为期望的速度向量
            ee_velocity_desired = torch.cat((ee_pos_vel, ee_rot_vel), dim=-1).unsqueeze(-1)

            # 根据当前关节角度和标称关节角度计算目标关节速度
            joint_vel_desired_by_pos = ((self.joint_pos_desired-joint_pos_current)@self.joint_pos_kp).unsqueeze(-1)

            # 对雅可比矩阵进行奇异值分解
            U, S, V = torch.svd(jacobian,some=False)
            # 求零空间（暂不考虑奇异点）
            # rtol = 1e-6
            # rank = (S > rtol * S[0]).sum(dim=1)
            rank = 6
            null_space = V[:,:,rank:]
            N = null_space.mT

            # 计算qp优化问题
            lambda_ = 0.01
            A = jacobian
            P = 2 * (torch.bmm(A.mT, A) + lambda_ * torch.bmm(N.mT, N))
            Q = -2 * torch.bmm(A.mT, ee_velocity_desired)
            - 2 * lambda_ * torch.bmm(torch.bmm(N.mT, N),joint_vel_desired_by_pos)

            # 使用雅可比矩阵的伪逆计算期望的关节速度
            joint_vel_desired = torch.linalg.lstsq(
                P, -Q
            ).solution.squeeze(2)
            # joint_vel_desired = torch.linalg.lstsq(
            #     jacobian, ee_velocity_desired
            # ).solution.squeeze(2)

            # 根据期望的关节速度计算期望的关节位置
            joint_pos_desired = joint_pos_current + joint_vel_desired * dt

            return {"joint_positions": joint_pos_desired}

        def set_goal(self, goal_pos, goal_ori):
            """
            设置目标位置和姿态。

            参数：
                goal_pos (torch.Tensor): 目标位置。
                goal_ori (torch.Tensor): 目标姿态（四元数）。
            """
            self.goal_pos = goal_pos
            self.goal_ori = goal_ori

        def reset(self):
            """重置控制器"""
            pass

    return DiffIKController(*args, **kwargs)
