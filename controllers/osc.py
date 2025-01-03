"""Code derived from https://github.com/StanfordVL/perls2 and https://github.com/ARISE-Initiative/robomimic"""
import math
from typing import Dict, List

import torch
import sys
import furniture_bench.controllers.control_utils as C


def osc_factory(real_robot=True, *args, **kwargs):
    """
    用于创建操作空间控制 (OSC) 控制器的工厂函数。

    Args:
        real_robot (bool): 如果控制器与真实的机器人一起使用，则为 True，否则为 False。
                           这决定了控制器的基类 (对于真实机器人为 torchcontrol.PolicyModule，
                           否则为 torch.nn.Module)。
        *args: 可变长度参数列表。
        **kwargs: 任意关键字参数。

    Returns:
        OSCController: OSCController 类的一个实例。
    """
    if real_robot:
        import torchcontrol as toco

        base = toco.PolicyModule
    else:
        base = torch.nn.Module

    class OSCController(base):
        """
        用于控制机械臂在任务空间中运动的操作空间控制 (OSC) 类。
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
            初始化末端执行器阻抗控制器。

            Args:
                kp (torch.Tensor): 用于根据位置/方向误差确定期望扭矩的位置增益。
                                    可以是标量（所有动作维度的值相同）或列表（每个维度都有特定的值）。
                kv (torch.Tensor): 用于根据速度/角速度误差确定期望扭矩的速度增益。
                                    可以是标量（所有动作维度的值相同）或列表（每个维度都有特定的值）。
                                    如果定义了 kv,则忽略阻尼。
                ee_pos_current (torch.Tensor): 当前末端执行器的位置。
                ee_quat_current (torch.Tensor): 当前末端执行器的方向。
                init_joints (torch.Tensor): 初始关节位置（用于零空间）。
                position_limits (torch.Tensor): 计算出的目标末端执行器位置大小将被限制在这些限制（米）之内和之上。
                                                可以是 2 元素列表（所有笛卡尔维度的最小/最大值相同）
                                                或 2 元素列表的列表（每个维度都有特定的最小/最大值）。
                mass_matrix_offset_val (list): 要添加到质量矩阵对角线最后三个元素的偏移量的 3f 列表。
                                                用于真实机器人，以调整末端关节处的高摩擦力。
                max_dx (float): 插值过程中位置移动的最大增量。
                control_freq (int): 控制循环的频率。
                policy_freq (int): 从机器人策略向此控制器馈送动作的频率。
                ramp_ratio (float): control_freq / policy_freq 的比率。用于确定插值器中要采取的步数。
                joint_kp (float): 关节位置控制的比例增益。
            """
            super().__init__()
            # limits
            # 末端执行器位置的限制
            self.position_limits = position_limits
            # 控制器增益
            self.kp = kp
            self.kv = kv
            # 初始关节位置
            self.init_joints = init_joints

            # 期望的末端执行器位置和方向（可以优化的参数）
            self.ee_pos_desired = torch.nn.Parameter(ee_pos_current)
            self.ee_quat_desired = torch.nn.Parameter(ee_quat_current)

            # 真实机器人的质量矩阵偏移值
            # self.mass_matrix = torch.zeros((7, 7))
            self.mass_matrix_offset_val = mass_matrix_offset_val
            self.mass_matrix_offset_idx = torch.tensor([[4, 4], [5, 5], [6, 6]])

            # 用于跟踪重复扭矩的变量
            self.repeated_torques_counter = 0
            self.num_repeated_torques = 3
            self.prev_torques = torch.zeros((7,))

            # 位置插值器参数
            # Interpolator pos, ori
            self.max_dx = max_dx  # Maximum allowed change per interpolator step
            self.total_steps = math.floor(
                ramp_ratio * float(controller_freq) / float(policy_freq)
            ) # 每个插值器动作的总步数

            # 用于插值的当前目标位置和先前目标位置
            self.goal_pos = ee_pos_current.clone()
            self.prev_goal_pos = ee_pos_current.clone()
            self.step_num_pos = 1

            # 方向插值器参数
            self.fraction = 0.5
            self.goal_ori = ee_quat_current.clone()
            self.prev_goal_ori = ee_quat_current.clone()
            self.step_num_ori = 1

            # 先前插值的位置和方向
            self.prev_interp_pos = ee_pos_current.clone()
            self.prev_interp_ori = ee_quat_current.clone()

            # 关节比例增益
            self.joint_kp = joint_kp

        def forward(
            self, state_dict: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            """
            控制器的正向传递，根据当前状态计算所需的关节扭矩。

            Args:
                state_dict (Dict[str, torch.Tensor]): 包含当前机器人状态的字典。 
                                                        应包含以下键：'joint_positions'、'joint_velocities'、 
                                                        'mass_matrix'、'ee_pose'、'jacobian'。

            Returns:
                Dict[str, torch.Tensor]: 包含计算出的关节扭矩（'joint_torques'）的字典。
            """

            # 增加重复扭矩计数器，并在必要时返回先前的扭矩
            self.repeated_torques_counter = (
                self.repeated_torques_counter + 1
            ) % self.num_repeated_torques
            if self.repeated_torques_counter != 1:
                return {"joint_torques": self.prev_torques}
            
            # 获取当前关节位置和速度
            joint_pos_current = state_dict["joint_positions"] # 始终为torch.Size([7])
            joint_vel_current = state_dict["joint_velocities"]

            # 获取质量矩阵并为真实机器人添加偏移量
            mass_matrix = state_dict["mass_matrix"].reshape(7, 7).t()
            mass_matrix[4, 4] += self.mass_matrix_offset_val[0]
            mass_matrix[5, 5] += self.mass_matrix_offset_val[1]
            mass_matrix[6, 6] += self.mass_matrix_offset_val[2]

            # 获取末端执行器位姿并提取位置和方向
            ee_pose = state_dict["ee_pose"].reshape(4, 4).t().contiguous()

            ee_pos, ee_quat = C.mat2pose(ee_pose)
            ee_pos = ee_pos.to(ee_pose.device)
            ee_quat = ee_quat.to(ee_pose.device)

            # 获取雅可比矩阵
            jacobian = state_dict["jacobian"].reshape(7, 6).t().contiguous()

            # 计算当前末端执行器的速度（线速度和角速度）
            ee_twist_current = jacobian @ joint_vel_current
            ee_pos_vel = ee_twist_current[:3]
            ee_ori_vel = ee_twist_current[3:]

            # 设置目标位置和方向，并在必要时对位置进行裁剪
            goal_pos = C.set_goal_position(self.position_limits, self.ee_pos_desired)
            goal_ori = self.ee_quat_desired

            # 设置用于插值的当前目标位置和方向
            self.set_goal(goal_pos, goal_ori)

            # 获取插值后的目标位置和方向
            goal_pos = self.get_interpolated_goal_pos()
            goal_ori = self.get_interpolated_goal_ori()

            # 将目标方向从四元数转换为旋转矩阵
            goal_ori_mat = C.quat2mat(goal_ori).to(goal_ori.device)
            ee_ori_mat = C.quat2mat(ee_quat).to(ee_quat.device)

            # 计算方向误差
            ori_error = C.orientation_error(goal_ori_mat, ee_ori_mat)

            # 使用控制律计算末端执行器处的期望力和扭矩
            position_error = goal_pos - ee_pos
            vel_pos_error = -ee_pos_vel
            desired_force = torch.multiply(
                position_error, self.kp[0:3]
            ) + torch.multiply(vel_pos_error, self.kv[0:3])

            vel_ori_error = -ee_ori_vel
            desired_torque = torch.multiply(ori_error, self.kp[3:]) + torch.multiply(
                vel_ori_error, self.kv[3:]
            )

            # 计算操作空间质量矩阵和零空间矩阵
            lambda_full, nullspace_matrix = C.opspace_matrices(mass_matrix, jacobian)

            # 计算期望的力/力矩（扳手）
            desired_wrench = torch.cat([desired_force, desired_torque])

            # 将扳手解耦为任务空间和零空间分量
            decoupled_wrench = torch.matmul(lambda_full, desired_wrench)

            # 将期望的扳手投影到关节扭矩上
            torques = torch.matmul(jacobian.T, decoupled_wrench) + C.nullspace_torques(
                mass_matrix,
                nullspace_matrix,
                self.init_joints,
                joint_pos_current,
                joint_vel_current,
                joint_kp=self.joint_kp,
            )

            # 应用扭矩偏移以防止机器人卡住
            self._torque_offset(ee_pos, goal_pos, torques)

            # 保存计算出的扭矩并返回
            self.prev_torques = torques

            return {"joint_torques": torques}

        def set_goal(self, goal_pos, goal_ori):
            """
            设置控制器的目标位置和方向。

            此方法更新目标位置和方向，并在设置了新目标或先前的插值完成时重置插值。

            Args:
                goal_pos (torch.Tensor): 末端执行器的期望目标位置。
                goal_ori (torch.Tensor): 末端执行器的期望目标方向。
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
            计算并返回插值后的目标位置。

            此方法根据当前位置、目标位置和插值步数计算下一个插值目标位置。

            Returns:
                torch.Tensor: 插值后的目标位置。
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
            使用球面线性插值 (slerp) 计算并返回插值后的目标方向。

            Returns:
                torch.Tensor: 插值后的目标方向。
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
            应用扭矩偏移以防止机器人在位置限制处卡住。

            此方法检查末端执行器是否位于位置限制处并远离目标移动。
            如果是，则应用扭矩偏移以帮助机器人向目标移动。

            Args:
                ee_pos (torch.Tensor): 当前末端执行器的位置。
                goal_pos (torch.Tensor): 期望的目标位置。
                torques (torch.Tensor): 计算出的关节扭矩。
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
