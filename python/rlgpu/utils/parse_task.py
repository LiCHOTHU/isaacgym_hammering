# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from rlgpu.tasks.cartpole import Cartpole
from rlgpu.tasks.cartpole_y_up import CartpoleYUp
from rlgpu.tasks.ball_balance import BallBalance
from rlgpu.tasks.ant import Ant
from rlgpu.tasks.humanoid import Humanoid
from rlgpu.tasks.franka import FrankaCabinet
from rlgpu.tasks.shadow_hand import ShadowHand
from rlgpu.tasks.franka_hammer import FrankaHammer
from rlgpu.tasks.base.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython

from isaacgym import rlgpu


def parse_task(args, cfg, cfg_train, sim_params):
    # create native task and pass custom config
    if args.task_type == "C++":
        task_cfg = cfg["env"]
        task_cfg["seed"] = cfg_train["seed"]
        if args.device == "CPU":
            print("C++ CPU")
            sim_device = 'cpu'
            if args.ppo_device == "GPU":
                ppo_device = 'cuda:0'
            else:
                ppo_device = 'cpu'
            task = rlgpu.create_task_cpu(args.task, json.dumps(task_cfg))
            if not task:
                warn_task_name()
            if args.headless:
                task.init(0, -1, args.physics_engine, sim_params)
            else:
                task.init(0, 0, args.physics_engine, sim_params)
            env = VecTaskCPU(task, ppo_device, False, cfg_train.get("clip_observations", 5.0), cfg_train.get("clip_actions", 1.0))
        elif args.device == "GPU":
            print("C++ GPU")
            sim_device = 'cuda:0'
            ppo_device = 'cuda:0'
            task = rlgpu.create_task_gpu(args.task, json.dumps(task_cfg))
            if not task:
                warn_task_name()
            if args.headless:
                task.init(0, -1, args.physics_engine, sim_params)
            else:
                task.init(0, 0, args.physics_engine, sim_params)
            env = VecTaskGPU(task, ppo_device, cfg_train.get("clip_observations", 5.0), cfg_train.get("clip_actions", 1.0))
    elif args.task_type == "Python":
        cfg["seed"] = cfg_train["seed"]
        if args.device == "CPU":
            print("Python CPU")
            sim_device = 'cpu'
            ppo_device = 'cuda:0' if args.ppo_device == "GPU" else 'cpu'
        else:
            print("Python GPU")
            sim_device = 'cuda:0'
            ppo_device = 'cuda:0'

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                graphics_device=-1 if args.headless else 0,
                device=sim_device)
        except NameError as e:
            warn_task_name()
        env = VecTaskPython(task, ppo_device)

    return task, env
