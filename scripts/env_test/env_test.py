"""
测试环境初始化（尤其是并行环境数目较多时）是否正常
原始环境中reset将会反复调用refresh导致机械臂发生移动
在非同步重启时影响将会更大
以及对reset_to和随机性进行测试
"""
import furniture_bench

import gym
import cv2
import torch
import numpy as np
from furniture_bench.envs.observation import (
    FULL_OBS,
    DEFAULT_VISUAL_OBS,
    DEFAULT_STATE_OBS,
)
from Common.geometry import isaac_quat_to_rot_6d, transform_rot_6d, isaac_quat_to_pytorch3d_quat
import copy

torch.set_printoptions(precision=4, linewidth=200)

def reset_test():
    num_envs=10
    # Create FurnitureSim environment.
    env = gym.make(
        "FurnitureSimRL-v0",
        # "FurnitureSimState-v0",
        furniture="one_leg",
        num_envs=num_envs,
        resize_img=True,
        init_assembled=False,
        record=False,
        headless=True,
        save_camera_input=False,
        randomness="low",
        high_random_idx=0,
        act_rot_repr="rot_6d",
        action_type = "pos",
        ctrl_mode = "diffik",
        compute_device_id=0,
        graphics_device_id=0,
        obs_keys=DEFAULT_STATE_OBS,
    )
    # Initialize FurnitureSim.
    ob = env.reset()
    init_ee_pos = ob['robot_state']['ee_pos']
    init_ee_rot_6d = isaac_quat_to_rot_6d(ob['robot_state']['ee_quat'])
    init_grasp = torch.zeros_like(ob['robot_state']['gripper_width'])
    init_action = torch.cat((init_ee_pos, init_ee_rot_6d, init_grasp), dim=1)

    print(ob)
    
    done = False

    for _ in range(5):
        ob, rew, done, _ = env.step(init_action)
    print("done")


def reset_some_test():
    num_envs=10
    # Create FurnitureSim environment.
    env = gym.make(
        # "FurnitureSimRL-v0",
        "FurnitureSimState-v0",
        furniture="one_leg",
        num_envs=num_envs,
        resize_img=True,
        init_assembled=False,
        record=False,
        headless=True,
        save_camera_input=False,
        randomness="low",
        high_random_idx=0,
        act_rot_repr="rot_6d",
        action_type = "pos",
        ctrl_mode = "diffik",
        compute_device_id=0,
        graphics_device_id=0,
        # obs_keys=DEFAULT_STATE_OBS,
    )
    # Initialize FurnitureSim.
    ob = env.reset()
    init_ee_pos = ob['robot_state']['ee_pos']
    init_ee_pos[:,0]= init_ee_pos[:,0]+0.1 #使其发生移动
    
    init_ee_rot_6d = isaac_quat_to_rot_6d(ob['robot_state']['ee_quat'])
    init_grasp = torch.zeros_like(ob['robot_state']['gripper_width'])
    init_action = torch.cat((init_ee_pos, init_ee_rot_6d, init_grasp), dim=1)
    done = False
    rest_index = torch.tensor([1,2,3,4,5])
    # rest_index = torch.tensor([1,2])
    for i in range(5):
        ob, rew, done, _ = env.step(init_action)
        print(i,ob['robot_state']['ee_pos'][:,0])
        if i==3:
            # rl环境
            # ob = env.reset(env_idxs=rest_index)
            # print(i,ob['robot_state']['ee_pos'][:,0])
            # sim环境
            for n in rest_index.tolist():
                env.reset_env(env_idx=n)
                env.refresh()
            
    env.close()
    print("done")

def reset_to_test():
    num_envs=10
    # Create FurnitureSim environment.
    env = gym.make(
        "FurnitureSimRL-v0",
        # "FurnitureSimState-v0",
        furniture="one_leg",
        num_envs=num_envs,
        resize_img=True,
        init_assembled=False,
        record=False,
        headless=True,
        save_camera_input=False,
        randomness="low",
        high_random_idx=0,
        act_rot_repr="rot_6d",
        action_type = "pos",
        ctrl_mode = "diffik",
        compute_device_id=0,
        graphics_device_id=0,
        obs_keys=DEFAULT_STATE_OBS,
    )
    # Initialize FurnitureSim.
    joint_positions = np.array([-0.0263,  0.3759,  0.1249, -2.1383, -0.0943,  2.4965,  0.0192])
    joint_velocities = np.array([0., 0., 0., 0., 0., 0., 0.])
    gripper_width = 0.065
    parts_poses = np.array([ 3.1830e-08,  2.4000e-01, -1.5685e-02,  3.8304e-09, -7.0711e-01,  7.0711e-01,  5.5557e-08, 
                                -2.0000e-01,  7.0000e-02, -1.5000e-02, -2.1896e-07, -7.0710e-01,  9.3661e-08,  7.0711e-01,
                                -1.2000e-01,  7.0000e-02, -1.5000e-02, -2.0157e-07, -7.0711e-01,  1.1921e-07,  7.0710e-01,  
                                1.2000e-01,  7.0000e-02, -1.5000e-02, -2.6098e-07, -7.0711e-01,  1.7881e-07,  7.0710e-01,
                                2.0000e-01,  7.0000e-02, -1.5000e-02, -6.2167e-07, -7.0711e-01,  2.9616e-07,  7.0711e-01,
                                2.3268e-17,  3.8000e-01, -1.5000e-02, -1.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00])
    robot_state = {
        "joint_positions":joint_positions,
        "joint_velocities":joint_velocities,
        "gripper_width":gripper_width,
    }
    state = {
        "robot_state":robot_state,
        "parts_poses":parts_poses,
    }
    init_states = [copy.deepcopy(state) for _ in range(num_envs)]
    for i in range(len(init_states)):
        init_states[i]["robot_state"]["joint_positions"][3] += i*0.02
    # print(init_states)

    env.reset()
    ob = env.reset_to(init_states)
    init_ee_pos = ob['robot_state']['ee_pos']
    init_ee_rot_6d = isaac_quat_to_rot_6d(ob['robot_state']['ee_quat'])
    init_grasp = torch.zeros_like(ob['robot_state']['gripper_width'])
    init_action = torch.cat((init_ee_pos, init_ee_rot_6d, init_grasp), dim=1)

    print(ob)
    
    done = False

    for _ in range(5):
        ob, rew, done, _ = env.step(init_action)
    print("done")


if __name__ == "__main__":
    # reset_test()
    """
    测试初始化得到的状态是否正常

    在FurnitureSimState-v0中若使用diffik需在reset()中使用
    self._reset_franka_all()
    self._reset_parts_all()
    替代循环初始化才能得到正常结果()
    (若使用osc且只进行全局初始化可以正常用)

    在FurnitureSimRL-v0中使用对RL环境重写的reset()而不是继承的即可
    """

    # reset_some_test()
    """
    测试对部分环境的初始化是否会导致其他环境的异常步进

    在FurnitureSimState-v0中原始代码不支持批量初始化单个环境,且每个环境初始化后需要refresh才会生效,
    故基本不能使用(可以重写)

    在FurnitureSimRL-v0中使用重写的reset时,每次重置会导致所有环境步进一步(仿真步,不是控制步)
    如果在reset时不进行refresh则不会有上述问题,但是被重置的环境将无法获得正确的观测量
    """

    reset_to_test()
    """
    测试对指定环境参数的初始化得到的状态是否正常

    在FurnitureSimState-v0中的reset_to是通过循环实现初始化的,一样有refresh的问题(可以重写)

    在FurnitureSimRL-v0中可正常使用
    """
    """
    若对部分环境初始化到指定状态,则：
    在FurnitureSimRL-v0中使用重写的reset_env_to时,每次重置会导致所有环境步进一步(仿真步,不是控制步)
    如果在reset时不进行refresh则不会有上述问题,但是被重置的环境将无法获得正确的观测量
    (不过由于环境初始化到指定状态,可以使用该指定状态作为初始观测量(注释refresh并在重构环境中实现即可))
    """



    



