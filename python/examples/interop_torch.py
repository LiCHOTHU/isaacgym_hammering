"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import numpy as np
import imageio

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch

use_viewer = True

gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(description="PyTorch tensor interop example")

# configure sim
sim_params = gymapi.SimParams()
sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 8
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.4
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

# This determines whether physics tensors are on CPU or GPU
sim_params.use_gpu_pipeline = True

sim = gym.create_sim(0, 0, args.physics_engine, sim_params)

ball_asset = gym.create_sphere(sim, 0.5, None)

# create viewer
if use_viewer:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()
else:
    viewer = None

gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_C, "write_camera_images")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_P, "print_state")

# Add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# set up env grid
num_envs = 4
envs_per_row = 2
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

dt = 0.01
frame_count = 0
t1 = 0

envs = []
cams = []
cam_tensors = []

# create envs
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 2.0, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor_handle = gym.create_actor(env, ball_asset, pose, "ball", i, 0)

    # set ball restitution
    props = gym.get_actor_rigid_shape_properties(env, actor_handle)
    props[0].restitution = 0.9
    gym.set_actor_rigid_shape_properties(env, actor_handle, props)

    # set ball color
    c = 0.5 + 0.5 * np.random.random(3)
    gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(c[0], c[1], c[2]))

    # add camera
    cam_props = gymapi.CameraProperties()
    cam_props.width = 128
    cam_props.height = 128
    cam_props.enable_tensors = True
    cam_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(cam_handle, env, gymapi.Vec3(5, 1, 0), gymapi.Vec3(0, 1, 0))
    cams.append(cam_handle)

    # obtain camera tensor
    cam_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_COLOR)
    print("Got camera tensor with shape", cam_tensor.shape)

    # wrap camera tensor in a pytorch tensor
    torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
    cam_tensors.append(torch_cam_tensor)
    print("  Torch camera tensor device:", torch_cam_tensor.device)
    print("  Torch camera tensor shape:", torch_cam_tensor.shape)

gym.prepare_sim(sim)

# get GPU physics state tensor
state_tensor = gym.acquire_rigid_body_state_tensor(sim)
print("Gym state tensor shape:", state_tensor.shape)
print("Gym state tensor data @ 0x%x" % state_tensor.data_address)

# wrap physics state tensor in a pytorch tensor
rb_states = gymtorch.wrap_tensor(state_tensor)
print("Torch state tensor device:", rb_states.device)
print("Torch state tensor shape:", rb_states.shape)
print("Torch state tensor data @ 0x%x" % rb_states.data_ptr())

# create some wrapper tensors for different slices
num_bodies = rb_states.shape[0]
rb_positions = gymtorch.wrap_tensor(state_tensor, counts=(num_bodies, 3))
rb_orientations = gymtorch.wrap_tensor(state_tensor, counts=(num_bodies, 4), offsets=(0, 3))
rb_linvels = gymtorch.wrap_tensor(state_tensor, counts=(num_bodies, 3), offsets=(0, 7))
rb_angvels = gymtorch.wrap_tensor(state_tensor, counts=(num_bodies, 3), offsets=(0, 10))

next_fps_report = 2.0

while viewer is None or not gym.query_viewer_has_closed(viewer):

    frame_no = gym.get_frame_count(sim)

    # must call simulate() before refresh_rigid_body_state_tensor()
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh state data in the tensor
    gym.refresh_rigid_body_state_tensor(sim)

    gym.step_graphics(sim)

    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    # process input actions
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "write_camera_images" and evt.value > 0:
            for i in range(num_envs):
                # use write_camera_image_to_file() API
                cam_handle = cams[i]
                fname = "cam-%04d-%04d-a.png" % (frame_no, i)
                gym.write_camera_image_to_file(sim, envs[i], cam_handle, gymapi.IMAGE_COLOR, fname)

                # write tensor to image
                fname2 = "cam-%04d-%04d-b.png" % (frame_no, i)
                cam_gpu_tensor = cam_tensors[i]
                cam_img = cam_gpu_tensor.cpu().numpy()
                imageio.imwrite(fname2, cam_img)

        if evt.action == "print_state" and evt.value > 0:
            print("========= Frame %d ==========" % frame_no)
            # print the state tensors
            print("RB states:")
            print(rb_states)
            print("RB positions:")
            print(rb_positions)
            print("RB orientations:")
            print(rb_orientations)
            print("RB linear velocities:")
            print(rb_linvels)
            print("RB angular velocities:")
            print(rb_angvels)
            print()

    t = gym.get_elapsed_time(sim)
    if t >= next_fps_report:
        t2 = gym.get_elapsed_time(sim)
        fps = frame_count / (t2 - t1)
        print("FPS %.1f (%.1f)" % (fps, fps * num_envs))
        frame_count = 0
        t1 = gym.get_elapsed_time(sim)
        next_fps_report = t1 + 2.0

    gym.end_access_image_tensors(sim)

    if viewer is not None:
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    frame_count += 1
