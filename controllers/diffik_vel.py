from typing import Dict

import torch

import furniture_bench.controllers.control_utils as C

import torch

from ipdb import set_trace as bp


def diffik_vel_factory(real_robot=True, *args, **kwargs):
    """
    创建微分逆运动学 (DiffIK) 速度控制器的factory函数。
    输入目标位置，输出关节速度。

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

    class DiffIKVelController(base):
        """Differential Inverse Kinematics Velocity Controller"""
        """微分逆运动学控制器"""

        def __init__(
            self,
            Kpos=torch.Tensor([6.0, 6.0, 6.0])*0.8,
            Krot=torch.Tensor([4.0, 4.0, 4.0])*0.8,
        ):
            """
            初始化微分逆运动学控制器。

            参数：
                Kp
                Kd
            """
            super().__init__()
            self.ee_pos_desired = None
            self.ee_quat_desired = None
            self.ee_pos_error = None
            self.ee_rot_error = None

            self.Kpos = Kpos
            self.Krot = Krot

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
            # joint_pos_current = state_dict["joint_positions"]  # 当前关节位置

            # jacobian 的形状: (batch_size, 6, num_joints = 7)
            jacobian = state_dict["jacobian_diffik"]

            # ee_pos 的形状: (batch_size, 3)
            # ee_quat 的形状: (batch_size, 4)，实部在末尾
            ee_pos, ee_quat_xyzw = state_dict["ee_pos"], state_dict["ee_quat"]  # 末端执行器位置和姿态（四元数）
            goal_ori_xyzw = self.goal_ori  # 目标姿态（四元数）

            position_error = self.goal_pos - ee_pos

            # 将四元数转换为旋转矩阵
            ee_mat = C.quaternion_to_matrix(ee_quat_xyzw)  # 当前末端执行器姿态的旋转矩阵
            goal_mat = C.quaternion_to_matrix(goal_ori_xyzw)  # 目标姿态的旋转矩阵

            # 计算矩阵误差
            mat_error = torch.matmul(goal_mat, torch.inverse(ee_mat))

            # 将矩阵误差转换为轴角表示
            ee_delta_axis_angle = C.matrix_to_axis_angle(mat_error)

            # 计算期望的末端执行器线速度和角速度
            ee_pos_vel = position_error * self.Kpos.to(position_error.device)  # 期望的线速度
            ee_rot_vel = ee_delta_axis_angle * self.Krot.to(position_error.device)  # 期望的角速度

            # 将期望的线速度和角速度合并为期望的速度向量
            ee_velocity_desired = torch.cat((ee_pos_vel, ee_rot_vel), dim=-1)

            # 使用雅可比矩阵的伪逆计算期望的关节速度
            # joint_vel_desired = torch.linalg.lstsq(
            #     jacobian, ee_velocity_desired
            # ).solution
            
            pinv = torch.linalg.pinv(jacobian)
            joint_vel_desired = torch.matmul(pinv, ee_velocity_desired.unsqueeze(-1)).squeeze(-1)

            return {"joint_velocity": joint_vel_desired}

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

    return DiffIKVelController(*args, **kwargs)
