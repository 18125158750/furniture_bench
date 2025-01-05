"""Instantiate FurnitureSimState-v0/FurnitureSimRL-v0 and test controllers."""
import gym
import cv2
import numpy as np
import furniture_bench.controllers.control_utils as C
from furniture_bench.envs.observation import (
    FULL_OBS,
    DEFAULT_VISUAL_OBS,
    DEFAULT_STATE_OBS,
)
from Common.geometry import isaac_quat_to_rot_6d, transform_rot_6d, isaac_quat_to_pytorch3d_quat
import pytorch3d.transforms as pt
import torch

def unwrap_angles(angles):
    """
    展开角度，避免周期性跳变。

    Args:
        angles: 角度列表 (弧度)。

    Returns:
        展开后的角度列表。
    """
    unwrapped_angles = [angles[0]]
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i-1]
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        unwrapped_angles.append(unwrapped_angles[-1] + diff)
    return unwrapped_angles

def test_position_response(env, dof_index, target_value, duration=60, initial_steps=20):
    """Tests the response of a specific position DOF (x, y, or z)."""

    assert 0 <= dof_index < 3, "dof_index must be 0, 1, or 2 for position control."

    ob = env.reset()
    init_ee_pos = ob['robot_state']['ee_pos']
    init_ee_rot_6d = isaac_quat_to_rot_6d(ob['robot_state']['ee_quat'])
    init_grasp = torch.zeros_like(ob['robot_state']['gripper_width'])
    init_action = torch.cat((init_ee_pos, init_ee_rot_6d, init_grasp), dim=1)

    measurements = []
    targets = []

    for _ in range(initial_steps):
        ob, _, _, _ = env.step(init_action)
        measurements.append(ob['robot_state']['ee_pos'][0, dof_index].cpu().numpy())
        targets.append(init_action[0, dof_index].cpu().numpy())

    init_action[0, dof_index] = init_action[0, dof_index] + target_value

    for _ in range(duration):
        ob, _, _, _ = env.step(init_action)
        measurements.append(ob['robot_state']['ee_pos'][0, dof_index].cpu().numpy())
        targets.append(init_action[0, dof_index].cpu().numpy())

    plot_response(measurements, targets, f"Position DOF {dof_index} Response")


def get_euler_from_obs(ob, dof_index):
    """Extracts Euler angle from observation for a given DOF."""
    quat = isaac_quat_to_pytorch3d_quat(ob['robot_state']['ee_quat'][0])  # Assuming you have this conversion function
    euler = pt.matrix_to_euler_angles(pt.quaternion_to_matrix(quat), "XYZ").cpu().numpy()
    return euler[dof_index - 3]

def get_euler_from_action(action, dof_index):
    """Extracts Euler angle from action for a given DOF."""
    euler = pt.matrix_to_euler_angles(pt.rotation_6d_to_matrix(action[0, 3:9]), "XYZ").cpu().numpy()
    return euler[dof_index - 3]

def set_target_rotation(action, dof_index, target_value):
    """Sets the target rotation in the action."""
    euler_angles = torch.zeros(1, 3).to(action.device)
    euler_angles[0, dof_index - 3] = target_value
    return transform_rot_6d(action[:, 3:9], euler_angles)

def test_orientation_response(env, dof_index, target_value, duration=60, initial_steps=20):
    """Tests the response of a specific orientation DOF (rotation around x, y, or z)."""

    assert 3 <= dof_index < 6, "dof_index must be 3, 4, or 5 for orientation control."

    ob = env.reset()
    init_ee_pos = ob['robot_state']['ee_pos']
    init_ee_rot_6d = isaac_quat_to_rot_6d(ob['robot_state']['ee_quat'])
    init_grasp = torch.zeros_like(ob['robot_state']['gripper_width'])
    init_action = torch.cat((init_ee_pos, init_ee_rot_6d, init_grasp), dim=1)

    measurements = []
    targets = []

    for _ in range(initial_steps):
        ob, _, _, _ = env.step(init_action)
        measurements.append(get_euler_from_obs(ob, dof_index))
        targets.append(get_euler_from_action(init_action, dof_index))

    init_action[:, 3:9] = set_target_rotation(init_action, dof_index, target_value)

    for _ in range(duration):
        ob, _, _, _ = env.step(init_action)
        measurements.append(get_euler_from_obs(ob, dof_index))
        targets.append(get_euler_from_action(init_action, dof_index))
    plot_response(unwrap_angles(measurements), unwrap_angles(targets), f"Orientation DOF {dof_index} Response")


def test_sine_wave_tracking(env, dof_index, amplitude, period, duration=200, initial_steps=20):
    """Tests the tracking performance for a sine wave input, using period instead of frequency."""

    assert 0 <= dof_index < 3, "dof_index must be 0, 1, or 2 for position control."

    ob = env.reset()
    init_ee_pos = ob['robot_state']['ee_pos']
    init_ee_rot_6d = isaac_quat_to_rot_6d(ob['robot_state']['ee_quat'])
    init_grasp = torch.zeros_like(ob['robot_state']['gripper_width'])
    init_action = torch.cat((init_ee_pos, init_ee_rot_6d, init_grasp), dim=1)

    initial_dof_value = init_action[0, dof_index].cpu().numpy() if dof_index < 3 else get_euler_from_action(init_action, dof_index)
    initial_rot_6d = init_action[:, 3:9].clone()  # Store initial rotation


    measurements = []
    targets = []

    for _ in range(initial_steps):
        ob, _, _, _ = env.step(init_action)
        measurements.append(ob['robot_state']['ee_pos'][0, dof_index].cpu().numpy() if dof_index < 3 else get_euler_from_obs(ob, dof_index))
        targets.append(init_action[0, dof_index].cpu().numpy() if dof_index < 3 else get_euler_from_action(init_action, dof_index))


    for i in range(duration):
        target_value = amplitude * np.sin(2 * np.pi * i / period)
        init_action[0, dof_index] = initial_dof_value + target_value  # Correct position control

        ob, _, _, _ = env.step(init_action)
        measurement = ob['robot_state']['ee_pos'][0, dof_index].cpu().numpy() if dof_index < 3 else get_euler_from_obs(ob, dof_index)
        target = initial_dof_value + target_value
        measurements.append(measurement)
        targets.append(target)
    plot_response(measurements, targets, f"Sine Wave Tracking DOF {dof_index}, Period: {period}")


def plot_response(measurements, targets, title):
    import matplotlib.pyplot as plt
    plt.figure(title)
    plt.plot(measurements, label="Measured")
    plt.plot(targets, label="Target", linestyle="--")
    plt.xlabel("Time Steps")
    plt.ylabel("DOF Value")
    plt.title(title)
    plt.legend()
    plt.show()

def main():
    num_envs = 1
    # Create FurnitureSim environment.
    env = gym.make(
        "FurnitureSimRL-v0",
        # "FurnitureSimState-v0",
        furniture="one_leg",
        num_envs=num_envs,
        resize_img=True,
        init_assembled=False,
        record=False,
        headless=False,
        save_camera_input=False,
        randomness="low",
        high_random_idx=0,
        act_rot_repr="rot_6d",
        action_type = "pos",
        ctrl_mode = "diffik-vel",
        # ctrl_mode = "diffik",
        compute_device_id=0,
        graphics_device_id=0,
        obs_keys=DEFAULT_STATE_OBS,
    )

    # test_sine_wave_tracking(env, dof_index=0, amplitude=0.1, period=10) # Test x position tracking

    test_position_response(env, dof_index=0, target_value=0.1)  # Test x position
    # test_position_response(env, dof_index=1, target_value=0.1)  # Test x position
    # test_position_response(env, dof_index=2, target_value=0.1)  # Test x position
    # test_orientation_response(env, dof_index=3, target_value=np.pi / 40) # Test x rotation (roll)
    # test_orientation_response(env, dof_index=4, target_value=np.pi / 40) # Test x rotation (roll)
    # test_orientation_response(env, dof_index=5, target_value=np.pi / 40) # Test x rotation (roll)


if __name__ == "__main__":
    main()
