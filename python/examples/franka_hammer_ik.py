"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Franka Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of Franka robot to pick up a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import matplotlib.pyplot as plt

import math
import numpy as np
import torch
import random
import time


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (2 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def compute_franka_reward(
    actions, device,
    hammer_pose, object_pose, object_linvel,
    franka_lfinger_pos, franka_lfinger_rot, franka_rfinger_pos, franka_rfinger_rot, down_dir,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]
    # how far the hand should be from hammer for grasping
    grasp_offset = 0.06
    gripper_close_offset = 0.025

    # set the reward weight and some configs here
    gripper_above_hammer_weight = 5.0
    gripper_to_hammer_dist_weight = 30.0
    two_finger_dist_reward_weight = 35.0
    gripper_downward_weight = 5.0
    gripper_hammer_y_weight = 3.0
    gripper_both_sides_weight = 5.0
    hammer_lift_weight = 10000.0
    hammer_init_height = 0.42
    hammer_max_height = 0.8

    # calculate grasp reward

    # calculate the gripper to hammer direction
    franka_gripper_pos = (franka_lfinger_pos + franka_rfinger_pos) / 2.0
    hammer_pos = hammer_pose[:, :3]

    to_hammer = (hammer_pos - franka_gripper_pos).to(device)
    gripper_hammer_dist = torch.norm(to_hammer, dim=-1).unsqueeze(-1)
    gripper_hammer_dir = to_hammer / gripper_hammer_dist

    gripper_hammer_dot = gripper_hammer_dir @ down_dir.view(3, 1)
    above_hammer_reward = (gripper_above_hammer_weight * (gripper_hammer_dot ** 2)*torch.sign(gripper_hammer_dot)).squeeze()


    # calculate the gripper to hammer distance
    d = torch.norm(franka_gripper_pos - hammer_pos, p=2, dim=-1).to(device)
    dist_reward = torch.where(d > grasp_offset, 1.0 - torch.tanh(d), to_torch([1.0]).repeat(num_envs).to(device))
    gripper_to_hammer_reward = dist_reward * gripper_to_hammer_dist_weight

    rewards = dist_reward * gripper_to_hammer_dist_weight
    rewards += above_hammer_reward

    # calculate the gripper towards direction should be facing downward
    gripper_rot = (franka_lfinger_rot + franka_rfinger_rot).to(device) / 2.0
    z_dir = to_torch([0, 0, 1]).repeat(num_envs, 1).to(device)
    gripper_z_dir = quat_apply(gripper_rot, z_dir)
    gripper_z_dot = gripper_z_dir @ down_dir.view(3, 1)
    gripper_downward_reward = (torch.sign(gripper_z_dot)*(gripper_z_dot**2) * gripper_downward_weight).squeeze()
    rewards += gripper_downward_reward

    # calculate the reward if the gripper_y and hammer_y alignment
    hammer_rot = hammer_pose[:, 3:7].to(device)
    y_dir = to_torch([0, 1, 0]).repeat(num_envs, 1).to(device)
    hammer_y_dir = quat_apply(hammer_rot, y_dir)
    gripper_y_dir = quat_apply(gripper_rot, y_dir)
    gripper_hammer_y_dot = torch.bmm(hammer_y_dir.view(num_envs, 1, 3), gripper_y_dir.view(num_envs, 3, 1))
    gripper_hammer_y_reward = ((gripper_hammer_y_dot ** 2) * gripper_hammer_y_weight).squeeze()
    rewards += gripper_hammer_y_reward

    # if gripper is close to hammer, add bonus for gripper on both sides of hammer
    hammer_to_lfinger = franka_lfinger_pos - hammer_pos
    hammer_to_lfinger_dist = torch.norm(hammer_to_lfinger, dim=-1).unsqueeze(-1)
    hammer_to_lfinger_dir = hammer_to_lfinger / hammer_to_lfinger_dist

    hammer_to_rfinger = franka_rfinger_pos - hammer_pos
    hammer_to_rfinger_dist = torch.norm(hammer_to_rfinger, dim=-1).unsqueeze(-1)
    hammer_to_rfinger_dir = hammer_to_rfinger / hammer_to_rfinger_dist

    hammer_both_sides_dot = torch.bmm(hammer_to_lfinger_dir.view(num_envs, 1, 3), hammer_to_rfinger_dir.view(num_envs, 3, 1)).squeeze()
    both_sides_reward = -torch.sign(hammer_both_sides_dot) * hammer_both_sides_dot ** 2

    rewards += both_sides_reward * gripper_both_sides_weight


    # add reward of distance of two fingers to hammer to encourage gripper close
    lfinger_dist = torch.norm(franka_lfinger_pos - hammer_pos, dim=-1)
    rfinger_dist = torch.norm(franka_rfinger_pos - hammer_pos, dim=-1)

    two_finger_dist = 2 - torch.tanh(lfinger_dist) - torch.tanh(rfinger_dist)

    # hand_hammer_close = (gripper_hammer_dist < grasp_offset).squeeze()
    two_finger_dist_reward = two_finger_dist_reward_weight * (two_finger_dist).squeeze()
    rewards += two_finger_dist_reward

    # determine if we're holding the hammer (grippers are closed and box is near)
    gripper_sep = torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim =-1).unsqueeze(-1).to(device)
    gripped = (gripper_sep < gripper_close_offset) & (gripper_hammer_dist < grasp_offset)

    # add penalty for dropping off the table
    hammer_drop_table = (hammer_pos[:, 2] < 0.4).float().squeeze().to(device)
    hammer_drop_penalty = -1000 * hammer_drop_table
    rewards += hammer_drop_penalty

    # add reward for lifting the hammer
    hammer_lift_height = torch.clamp(hammer_pos[:, 2] - hammer_init_height, 0, hammer_max_height).to(device)
    hammer_lift_reward = gripped.float().squeeze() * hammer_lift_height * hammer_lift_weight
    rewards += hammer_lift_reward

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions.squeeze() ** 2, dim=-1).to(device)
    rewards -= action_penalty_scale * action_penalty

    return rewards, above_hammer_reward, gripper_to_hammer_reward, gripper_downward_reward, gripper_hammer_y_reward, hammer_lift_reward, both_sides_reward, two_finger_dist_reward


# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Franka Jacobian Inverse Kinematics Example")

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.physx_gpu
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# set torch device
# device = 'cuda'
device = 'cuda' if sim_params.use_gpu_pipeline else 'cpu'

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = "../../assets"

# create table asset
table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)


# create box asset
box_size = 0.03
box_dims = gymapi.Vec3(0.04, 0.04, 0.04)
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, box_dims.x, box_dims.y, box_dims.z, asset_options)

'''
# create hammer asset
box_asset_file = "urdf/hammer_convex.urdf"
asset_options = gymapi.AssetOptions()
box_size = 0.035
box_asset = gym.load_asset(sim, asset_root, box_asset_file, asset_options)
'''

box_size = 0.06
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, box_size, box_size, box_size * 2, asset_options)

# load franka asset
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# configure franka dofs
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_lower_limits = franka_dof_props["lower"]
franka_upper_limits = franka_dof_props["upper"]
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

# use position drive for all dofs
franka_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][:7].fill(400.0)
franka_dof_props["damping"][:7].fill(40.0)
# grippers
franka_dof_props["stiffness"][7:].fill(800.0)
franka_dof_props["damping"][7:].fill(40.0)

# default dof states and position targets
franka_num_dofs = gym.get_asset_dof_count(franka_asset)
default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)
default_dof_pos[:7] = franka_mids[:7]
# grippers open
default_dof_pos[7:] = franka_upper_limits[7:]

default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# get link index of panda hand, which we will use as end effector
franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)
franka_hand_index = franka_link_dict["panda_hand"]

table_dof = gym.get_asset_rigid_body_count(table_asset)
box_dof = gym.get_asset_rigid_body_count(box_asset)


# configure env grid
num_envs = 1
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0, 0, 0.1)
franka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(-0.5, 0.0, 0.5 * table_dims.z)

box_pose = gymapi.Transform()

envs = []
box_idxs = []
hand_idxs = []
init_pos_list = []
init_rot_list = []
hammer_actor_idxs = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # add box
    box_pose.p.x = table_pose.p.x
    box_pose.p.y = table_pose.p.y
    box_pose.p.z = table_dims.z + box_size
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # get hammer actor index
    hammer_actor_idx = gym.get_actor_index(env, box_handle, gymapi.DOMAIN_SIM)
    hammer_actor_idxs.append(hammer_actor_idx)

    # get global index of box in rigid body state tensor
    box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
    box_idxs.append(box_idx)

    # add franka
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 2)


    # set dof properties
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

    # get initial hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

    # save left and right finger handle
    lfinger_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_leftfinger")
    rfinger_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_rightfinger")


# point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping
down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# box corner coords, used to determine grasping yaw
box_half_size = 0.5 * box_size
corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
corners = torch.stack(num_envs * [corner_coord]).to(device)

# downward axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)


# jacobian entries corresponding to franka hand
j_eef = jacobian[:, franka_hand_index - 1, :]

# Actor root state tensor
_actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
actor_root_state_tensor = gymtorch.wrap_tensor(_actor_root_state_tensor).view(-1, 13)

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 9, 1)

# Create a tensor noting whether the hand should return to the initial position
hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

# save some reward list
rewards = []
gripper_hammer_dir_rewards = []
gripper_hammer_dis_rewards = []
gripper_down_rewards = []
gripper_hammer_y_rewards = []
hammer_lift_rewards = []
both_side_rewards = []
two_finger_dist_rewards = []

itr = 0

# simulation loop
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_actor_root_state_tensor(sim)

    box_pos = rb_states[box_idxs, :3]
    box_rot = rb_states[box_idxs, 3:7]
    
    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]

    to_box = box_pos - hand_pos
    box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    box_dir = to_box / box_dist
    box_dot = box_dir @ down_dir.view(3, 1)

    # get pose for hammer, box and finger
    hammer_pose = actor_root_state_tensor[hammer_actor_idxs, 0:7]
    hammer_pos = actor_root_state_tensor[hammer_actor_idxs, 0:3]
    hammer_rot = actor_root_state_tensor[hammer_actor_idxs, 3:7]
    hammer_linvel = actor_root_state_tensor[hammer_actor_idxs, 7:10]
    hammer_angvel = actor_root_state_tensor[hammer_actor_idxs, 10:13]

    rb_states_reshape = rb_states.view(num_envs, -1, 13)

    franka_lfinger_pos = rb_states_reshape[:, lfinger_handle][:, 0:3]
    franka_rfinger_pos = rb_states_reshape[:, rfinger_handle][:, 0:3]
    franka_lfinger_rot = rb_states_reshape[:, lfinger_handle][:, 3:7]
    franka_rfinger_rot = rb_states_reshape[:, rfinger_handle][:, 3:7]

    # how far the hand should be from box for grasping
    grasp_offset = 0.1

    # determine if we're holding the box (grippers are closed and box is near)
    gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]
    gripped = (gripper_sep < box_size) & (box_dist < grasp_offset + box_size)

    yaw_q = cube_grasping_yaw(box_rot, corners)
    box_yaw_dir = quat_axis(yaw_q, 0)
    hand_yaw_dir = quat_axis(hand_rot, 0)
    yaw_dot = torch.bmm(box_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

    # determine if we have reached the initial position; if so allow the hand to start moving to the box
    to_init = init_pos - hand_pos
    init_dist = torch.norm(to_init, dim=-1)
    hand_restart = (hand_restart & (init_dist > 0.02)).squeeze(-1)
    return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)

    # if hand is above box, descend to grasp offset
    # otherwise, seek a position above the box
    above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset * 3)).squeeze(-1)
    grasp_pos = box_pos.clone()
    grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2] + grasp_offset + box_size * 0.5, box_pos[:, 2] + grasp_offset * 2.5 + box_size * 0.5)

    # compute goal position and orientation
    goal_pos = torch.where(return_to_start, init_pos, grasp_pos)
    goal_rot = torch.where(return_to_start, init_rot, quat_mul(down_q, quat_conjugate(yaw_q)))

    # compute position and orientation error
    pos_err = goal_pos - hand_pos
    orn_err = orientation_error(goal_rot, hand_rot)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    d = 0.05  # damping term
    lmbda = torch.eye(6).to(device) * (d ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 9, 1)

    # update position targets
    pos_target = dof_pos + u

    # gripper actions depend on distance between hand and box
    close_gripper = (box_dist < grasp_offset + 0.025) | gripped
    # always open the gripper above a certain height, dropping the box and restarting from the beginning
    hand_restart = hand_restart | (box_pos[:, 2] > 0.7)
    keep_going = torch.logical_not(hand_restart)
    close_gripper = close_gripper & keep_going.unsqueeze(-1)
    grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * num_envs).to(device), torch.Tensor([[0.04, 0.04]] * num_envs).to(device))
    pos_target[:, 7:9] = grip_acts.unsqueeze(-1)

    # set new position targets
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_target))

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

    print("pos: ", pos_target)
    print("box_pos: ", box_pos)
    print("box_rot: ", box_rot)

    # check reward plot
    down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)
    dist_reward_scale = 1.5
    rot_reward_scale = 0.5
    around_handle_reward_scale = 1.0
    open_reward_scale = 4.0
    finger_dist_reward_scale = 10.0
    action_penalty_scale = 7.5
    distX_offset = 0.04
    max_episode_length = 500

    # compute the reward after this state
    reward, above_hammer_reward, gripper_to_hammer_reward, gripper_downward_reward, gripper_hammer_y_reward, hammer_lift_reward, both_side_reward, two_finger_dist_reward = \
        compute_franka_reward(
            u, device,
            hammer_pose, None, None,
            franka_lfinger_pos, franka_lfinger_rot, franka_rfinger_pos, franka_rfinger_rot, down_dir,
            num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
            finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
        )

    rewards.append(float(reward.squeeze().cpu().numpy()))
    gripper_hammer_dir_rewards.append(above_hammer_reward.squeeze().cpu().numpy())
    gripper_hammer_dis_rewards.append(gripper_to_hammer_reward.squeeze().cpu().numpy())
    gripper_down_rewards.append(gripper_downward_reward.squeeze().cpu().numpy())
    gripper_hammer_y_rewards.append(gripper_hammer_y_reward.squeeze().cpu().numpy())
    hammer_lift_rewards.append(hammer_lift_reward.squeeze().cpu().numpy())
    both_side_rewards.append(both_side_reward.squeeze().cpu().numpy())
    two_finger_dist_rewards.append(two_finger_dist_reward.squeeze().cpu().numpy())

    itr += 1
    print(itr)

'''
plt.plot(range(199), rewards[1:])
plt.xlabel('Iteration')
plt.ylabel('Mean Reward')
plt.savefig("reward_plot.png")
plt.clf()

plt.plot(range(199), gripper_hammer_dir_rewards[1:])
plt.xlabel('Iteration')
plt.ylabel('Gripper Hammer dir Reward')
plt.savefig("gripper_hammer_dir_plot.png")
plt.clf()

plt.plot(range(199), gripper_hammer_dis_rewards[1:])
plt.xlabel('Iteration')
plt.ylabel('Gripper Hammer dis Reward')
plt.savefig("gripper_hammer_dis_plot.png")
plt.clf()

plt.plot(range(199), gripper_down_rewards[1:])
plt.xlabel('Iteration')
plt.ylabel('Gripper Downward Reward')
plt.savefig("gripper_down_plot.png")
plt.clf()

plt.plot(range(199), gripper_hammer_y_rewards[1:])
plt.xlabel('Iteration')
plt.ylabel('Gripper Y Reward')
plt.savefig("gripper_y_plot.png")
plt.clf()

plt.plot(range(199), hammer_lift_rewards[1:])
plt.xlabel('Iteration')
plt.ylabel('Hammer lift Reward')
plt.savefig("hammer_lift_plot.png")
plt.clf()

plt.plot(range(199), both_side_rewards[1:])
plt.xlabel('Iteration')
plt.ylabel('Both Side Reward')
plt.savefig("both_side_plot.png")
plt.clf()

plt.plot(range(199), two_finger_dist_rewards[1:])
plt.xlabel('Iteration')
plt.ylabel('Two finger dist Reward')
plt.savefig("two_finger_dist_plot.png")
plt.clf()

'''
# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


