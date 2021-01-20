"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Apply Force (apply_force.py)
----------------------------
This example shows how to apply a force onto a rigid body using the tensor API.
A force of 50 Newtons is applied onto alternating X, Y, and Z axis every 100 steps.

The ``--device`` option can be used to specify whether force tensors reside on CPU or GPU.
"""

from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym import gymtorch

import numpy as np
import torch

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Example of applying forces to bodies",
    custom_parameters=[
        {"name": "--device", "type": str, "default": "GPU", "help": "Device for force buffer - CPU or GPU"}])

# configure sim
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

if args.device == "GPU":
    sim_params.use_gpu_pipeline = True
    device = 'cuda:0'
else:
    device = 'cpu'

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load ball asset
asset_root = "../../assets"
asset_file = "urdf/ball.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())
num_bodies = gym.get_asset_rigid_body_count(asset)
print('num_bodies', num_bodies)

# default pose
pose = gymapi.Transform()
pose.p.y = 1.0

# set up the env grid
num_envs = 9
num_per_row = int(np.sqrt(num_envs))
env_spacing = 1.25
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# set random seed
np.random.seed(17)

envs = []
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # generate random bright color
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])

    ahandle = gym.create_actor(env, asset, pose, None, i, 1)
    gym.set_rigid_body_color(env, ahandle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

gym.prepare_sim(sim)

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 5, 20), gymapi.Vec3(0, 1, 0))

frame_count = 0
force_axis = 0
while not gym.query_viewer_has_closed(viewer):
    # apply force every 100 steps
    if frame_count % 100 == 0:
        force = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)
        force_axis = force_axis % 3
        force[:, 0, force_axis] = 50.0  # apply force in force_axis direction on root
        print("applying force in axis {}".format(force_axis))
        force_axis += 1
        force_tensor = gymtorch.unwrap_tensor(force)
        gym.apply_rigid_body_force_tensor(sim, force_tensor)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

    frame_count += 1

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
