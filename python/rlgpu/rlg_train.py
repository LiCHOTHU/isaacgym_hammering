# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg, retrieve_cfg
from utils.parse_task import parse_task

from rl_games.common import env_configurations, experiment, vecenv
from rl_games.torch_runner import Runner

import numpy as np
import copy
import torch


def create_rlgpu_env(**kwargs):
    sim_params = parse_sim_params(args, cfg, cfg_train)
    task, env = parse_task(args, cfg, cfg_train, sim_params)

    print(env.num_envs)
    print(env.num_obs)
    print(env.num_actions)

    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.obs = self.env.reset()

    def step(self, action):
        next_state, reward, is_done, info = self.env.step(action)

        return next_state, reward, is_done, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        print(info['action_space'], info['observation_space'])

        return info


vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU'})


if __name__ == '__main__':
    # Create default directories for weights and statistics
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    set_np_formatting()
    args = get_args(use_rlg_config=True)
    cfg, cfg_train, logdir = load_cfg(args, use_rlg_config=True)
    sim_params = parse_sim_params(args, cfg, cfg_train)

    set_seed(cfg_train["seed"])
    vargs = vars(args)

    runner = Runner()
    runner.load(cfg_train)
    runner.reset()
    runner.run(vargs)
