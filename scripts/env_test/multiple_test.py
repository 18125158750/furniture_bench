"""Instantiate FurnitureSim-v0 and test various functionalities."""

import argparse
import pickle

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


def main():
    num_envs=1000
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
        action_type = "delta",
        ctrl_mode = "diffik",
        compute_device_id=0,
        graphics_device_id=0,
        obs_keys=DEFAULT_STATE_OBS,
    )
    # Initialize FurnitureSim.
    ob = env.reset()
    # print(ob)
    
    done = False

    # 将动作转换为张量
    def action_tensor(ac):
        if isinstance(ac, (list, np.ndarray)):
            return torch.tensor(ac).float().to(env.device)

        ac = ac.clone()
        if len(ac.shape) == 1:
            ac = ac[None]
        return ac.tile(num_envs, 1).float().to(env.device)


    # Execute randomly sampled actions.
    import tqdm

    pbar = tqdm.tqdm()
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(1000000//num_envs):
        ac = action_tensor(env.action_space.sample())
        ob, rew, done, _ = env.step(ac)
        pbar.update(num_envs)
    # print(rew)
    # print(done)
    # print(ob)
    profiler.disable()
    print("result:")
    profiler.print_stats(sort='time')
    # profiler.dump_stats("pipeline.prof")

    print("done")


if __name__ == "__main__":
    main()
