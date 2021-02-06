"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Franka Operational Space Control
----------------
Operational Space Control of Franka robot to demonstrate Jacobian and Mass Matrix Tensor APIs
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import matplotlib.pyplot as plt

import math
import numpy as np
import torch


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def compute_franka_reward(
    actions, device,
    hammer_pose, object_pose, object_linvel,
    franka_lfinger_pos, franka_lfinger_rot, franka_rfinger_pos, franka_rfinger_rot, down_dir,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    # how far the hand should be from hammer for grasping
    grasp_offset = 0.02
    gripper_close_offset = 0.025

    # set the reward weight and some configs here
    gripper_above_hammer_weight = 1.0
    gripper_to_hammer_dist_weight = 25.0
    gripper_downward_weight = 1.0
    gripper_hammer_y_weight = 1.0
    gripper_both_sides_bonus = 0.0
    gripper_grasp_hammer_bonus = 0.0
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
    above_hammer_reward = gripper_above_hammer_weight * (gripper_hammer_dot ** 2) * torch.sign(gripper_hammer_dot)
    rewards = above_hammer_reward

    # calculate the gripper to hammer distance
    d = torch.norm(franka_gripper_pos - hammer_pos, p=2, dim=-1).to(device)
    dist_reward = torch.where(d > grasp_offset, 1.0 - torch.tanh(d), to_torch([1.0]).repeat(num_envs))

    gripper_to_hammer_reward = dist_reward * gripper_to_hammer_dist_weight

    rewards += gripper_to_hammer_reward

    # calculate the gripper towards direction should be facing downward
    gripper_rot = (franka_lfinger_rot + franka_rfinger_rot).to(device) / 2.0
    z_dir = to_torch([0, 0, 1]).repeat(num_envs, 1).to(device)
    gripper_z_dir = quat_apply(gripper_rot, z_dir)
    gripper_z_dot = gripper_z_dir @ down_dir.view(3, 1)
    gripper_downward_reward = (torch.sign(gripper_z_dot) * ((gripper_z_dot)**2) * gripper_downward_weight).squeeze()
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
    # TODO: Shouldn't have this reward since it cannot fit for random rotaion of the hammer
    gripper_close_hammer = (gripper_hammer_dist < grasp_offset).squeeze()

    gripper_rr_hammer = torch.where((franka_rfinger_pos[:, 1].to(device) > hammer_pos[:, 1].to(device)), True, False).to(device)
    gripper_rl_hammer = torch.where((franka_rfinger_pos[:, 1].to(device) < hammer_pos[:, 1].to(device)), True, False).to(device)
    gripper_lr_hammer = torch.where((franka_lfinger_pos[:, 1].to(device) > hammer_pos[:, 1].to(device)), True, False).to(device)
    gripper_ll_hammer = torch.where((franka_lfinger_pos[:, 1].to(device) < hammer_pos[:, 1].to(device)), True, False).to(device)

    gripper_both_sides = ((gripper_rr_hammer & gripper_ll_hammer) | (gripper_rl_hammer & gripper_lr_hammer)) & gripper_close_hammer

    gripper_both_sides_reward = gripper_both_sides.float().squeeze() * gripper_both_sides_bonus
    rewards += gripper_both_sides_reward

    # check contact

    # determine if we're holding the hammer (grippers are closed and box is near)
    gripper_sep = torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim =-1).unsqueeze(-1).to(device)
    gripped = ((gripper_sep < gripper_close_offset) & (gripper_hammer_dist < grasp_offset)).squeeze() & gripper_both_sides
    gripper_grasp_hammer_reward = gripper_grasp_hammer_bonus * (gripped.float().squeeze())
    rewards += gripper_grasp_hammer_reward

    # add penalty for dropping off the table
    hammer_drop_table = (hammer_pos[:, 2] < 0.4).float().squeeze().to(device)
    hammer_drop_penalty = -1000 * hammer_drop_table
    rewards += hammer_drop_penalty

    # add reward for lifting the hammer
    gripped2 = ((gripper_sep < gripper_close_offset) & (gripper_hammer_dist < grasp_offset)).squeeze()
    hammer_lift_height = torch.clamp(hammer_pos[:, 2] - hammer_init_height, 0, hammer_max_height - hammer_init_height).to(device)
    hammer_lift_reward = torch.tanh(hammer_lift_height) * hammer_lift_weight * gripped2.float()
    rewards += hammer_lift_reward

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions.squeeze() ** 2, dim=-1).to(device)
    rewards -= action_penalty_scale * action_penalty

    return rewards, above_hammer_reward, gripper_to_hammer_reward, gripper_downward_reward, gripper_hammer_y_reward, hammer_lift_reward



# Parse arguments
args = gymutil.parse_arguments(description="Franka Tensor OSC Example",
                               custom_parameters=[
                                   {"name": "--pos_control", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Trace circular path in XZ plane"},
                                   {"name": "--orn_control", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Send random orientation commands"}])

# Initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.physx.contact_offset = 0.005
else:
    raise Exception("This example can only be used with PhysX")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    raise Exception("Failed to create sim")

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

device = 'cuda'

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)


asset_root = "../../assets"
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
hammer_asset_file = "urdf/hammer.urdf"

# create table asset
table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create hammer asset
asset_options = gymapi.AssetOptions()
asset_options.density = 200
hammer_asset = gym.load_asset(sim, asset_root, hammer_asset_file, asset_options)

# create box asset
box_size = 0.06
asset_options = gymapi.AssetOptions()
asset_options.density = 200
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

# Load franka asset
asset_options = gymapi.AssetOptions()
asset_options.flip_visual_attachments = True
asset_options.fix_base_link = True
asset_options.collapse_fixed_joints = True
asset_options.disable_gravity = True
asset_options.thickness = 0.001
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=device)
franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=device)


# get joint limits and ranges for Franka
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_lower_limits = franka_dof_props['lower']
franka_upper_limits = franka_dof_props['upper']
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
franka_num_dofs = len(franka_dof_props)


# set default DOF states
default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"][:7] = franka_mids[:7]

# set DOF control properties (except grippers)
franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
franka_dof_props["stiffness"][:7].fill(0.0)
franka_dof_props["damping"][:7].fill(0.0)

# set DOF control properties for grippers
franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][7:].fill(800.0)
franka_dof_props["damping"][7:].fill(40.0)


# Set up the env grid
num_envs = 1
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# default franka pose
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

print("Creating %d environments" % num_envs)

envs = []
hand_idxs = []
init_pos_list = []
init_orn_list = []
hammer_idxs = []
hammer_actor_idxs = []
box_idxs = []
hammer_init_state = []
box_init_state = []
box_actor_idxs = []


for i in range(num_envs):
    # Create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # Add franka
    franka_handle = gym.create_actor(env, franka_asset, pose, "franka", i, 1)

    # Set initial DOF states
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # Set DOF control properties
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # Get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_link7")

    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_orn_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # Get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_link7", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

    # set franka dof properties
    franka_dof_props = gym.get_asset_dof_properties(franka_asset)
    franka_dof_lower_limits = []
    franka_dof_upper_limits = []

    franka_dof_props['effort'][7] = 200
    franka_dof_props['effort'][8] = 200

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(-0.5, 0.0, 0.5 * table_dims.z)

    hammer_pose = gymapi.Transform()
    box_pose = gymapi.Transform()


    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)
    table_idx = gym.get_actor_rigid_body_index(env, table_handle, 0, gymapi.DOMAIN_SIM)

    # add hammer
    # avoid random position
    hammer_pose.p.x = table_pose.p.x + 0.
    hammer_pose.p.y = table_pose.p.y - 0.1
    # hammer_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
    # hammer_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
    hammer_pose.p.z = table_dims.z + 0.5 * box_size
    # hammer_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    # avoid random rotation
    hammer_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -math.pi / 2)

    hammer_actor = gym.create_actor(env, hammer_asset, hammer_pose, "hammer", i, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, hammer_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # get the global index of hammer in rigid body state tensor
    hammer_idx = gym.get_actor_rigid_body_index(env, hammer_actor, 0, gymapi.DOMAIN_SIM)
    hammer_idxs.append(hammer_idx)

    # get hammer actor index
    hammer_actor_idx = gym.get_actor_index(env, hammer_actor, gymapi.DOMAIN_SIM)
    hammer_actor_idxs.append(hammer_actor_idx)

    # set hammer initial state
    hammer_init_state.append([hammer_pose.p.x, hammer_pose.p.y, hammer_pose.p.z,
                                   hammer_pose.r.x, hammer_pose.r.y, hammer_pose.r.z, hammer_pose.r.w,
                                   0, 0, 0, 0, 0, 0])

    # add box
    box_relative_pose = gymapi.Transform()
    box_relative_pose.p = gymapi.Vec3(-0.2, 0.2, 0)
    box_pose.p.x = table_pose.p.x + box_relative_pose.p.x
    box_pose.p.y = table_pose.p.y + box_relative_pose.p.y
    box_pose.p.z = table_dims.z + 0.5 * box_size
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi)
    box_actor = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))

    # set box initial state
    box_init_state.append([box_pose.p.x, box_pose.p.y, box_pose.p.z,
                                box_pose.r.x, box_pose.r.y, box_pose.r.z, box_pose.r.w,
                                0, 0, 0, 0, 0, 0])

    # get the global index of box in rigid body state tensor
    box_idx = gym.get_actor_rigid_body_index(env, box_actor, 0, gymapi.DOMAIN_SIM)
    box_idxs.append(box_idx)

    # get box actor index
    box_actor_idx = gym.get_actor_index(env, box_actor, gymapi.DOMAIN_SIM)
    box_actor_idxs.append(box_actor_idx)

    gym.set_rigid_body_color(env, box_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    print("initialize: ", i)

    lfinger_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_leftfinger")
    rfinger_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_rightfinger")

hammer_actor_idxs = to_torch(hammer_actor_idxs, dtype=torch.long, device=device)
box_actor_idxs = to_torch(box_actor_idxs, dtype=torch.long, device=device)


# Point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 3)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API to access and control the physics simulation
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3)
init_orn = torch.Tensor(init_orn_list).view(num_envs, 4)

# desired hand positions and orientations
pos_des = init_pos.clone()
orn_des = init_orn.clone()

# Prepare jacobian tensor
# For franka, tensor shape is (num_envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)

# Jacobian entries for end effector
hand_index = gym.get_asset_rigid_body_dict(franka_asset)["panda_link7"]
j_eef = jacobian[:, hand_index - 1, :]

# Prepare mass matrix tensor
# For franka, tensor shape is (num_envs, 9, 9)
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
mm = gymtorch.wrap_tensor(_massmatrix)

kp = 5
kv = 2 * math.sqrt(kp)

# Rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# Actor root state tensor
_actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
actor_root_state_tensor = gymtorch.wrap_tensor(_actor_root_state_tensor).view(-1, 13)


# DOF state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_vel = dof_states[:, 1].view(num_envs, -1, 1)
dof_pos = dof_states[:, 0].view(num_envs, -1, 1)

itr = 0

# Here we set some action steps

control = {}
control['pos'] = [torch.Tensor([[-0.45, -0.1, 0.83]]*num_envs), torch.Tensor([[-0.45, -0.1, 0.54]]*num_envs),
                 torch.Tensor([[-0.45, -0.1, 0.54]]*num_envs), torch.Tensor([[-0.45, -0.1, 0.83]]*num_envs),]
control['orn'] = [torch.Tensor([[0, 1, 0, 0]]*num_envs) for i in range(4)]
control['close'] = [torch.Tensor([[0.08, 0.08]] * num_envs), torch.Tensor([[0.08, 0.08]] * num_envs),
                   torch.Tensor([[0.0, 0.0]] * num_envs), torch.Tensor([[0.0, 0.0]] * num_envs)]

itr_step = 200
rewards = []
gripper_hammer_dir_rewards = []
gripper_hammer_dis_rewards = []
gripper_down_rewards = []
gripper_hammer_y_rewards = []
hammer_lift_rewards = []

while not gym.query_viewer_has_closed(viewer):
    contacts = gym.get_env_rigid_contacts(envs[0])
    # if len(contacts) > 0 and itr == 300:
        # import pdb; pdb.set_trace()

    # Randomize desired hand orientations
    if itr % 250 == 0 and args.orn_control:
        orn_des = torch.rand_like(orn_des)
        orn_des /= torch.norm(orn_des)

    if itr >= 500 and itr < 550:
        force = gymapi.Vec3(0.0, 0.0, 3.0)
        pos = gymapi.Vec3(0.0, 0.0, 0.0)
        gym.apply_body_force(envs[0], hammer_idx, force, pos)


    itr += 1
    
    # get pose for hammer, box and finger 
    hammer_pose = actor_root_state_tensor[hammer_actor_idxs, 0:7]
    hammer_pos = actor_root_state_tensor[hammer_actor_idxs, 0:3]
    hammer_rot = actor_root_state_tensor[hammer_actor_idxs, 3:7]
    hammer_linvel = actor_root_state_tensor[hammer_actor_idxs, 7:10]
    hammer_angvel = actor_root_state_tensor[hammer_actor_idxs, 10:13]

    box_pose = actor_root_state_tensor[box_actor_idxs, 0:7]
    box_pos = actor_root_state_tensor[box_actor_idxs, 0:3]
    box_rot = actor_root_state_tensor[box_actor_idxs, 3:7]
    box_linvel = actor_root_state_tensor[box_actor_idxs, 7:10]
    box_angvel = actor_root_state_tensor[box_actor_idxs, 10:13]

    rb_states_reshape = rb_states.view(num_envs, -1, 13)

    franka_lfinger_pos = rb_states_reshape[:, lfinger_handle][:, 0:3]
    franka_rfinger_pos = rb_states_reshape[:, rfinger_handle][:, 0:3]
    franka_lfinger_rot = rb_states_reshape[:, lfinger_handle][:, 3:7]
    franka_rfinger_rot = rb_states_reshape[:, rfinger_handle][:, 3:7]

    # Update jacobian and mass matrix
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_actor_root_state_tensor(sim)

    # Get current hand poses
    pos_cur = rb_states[hand_idxs, :3]
    orn_cur = rb_states[hand_idxs, 3:7]

    # set the eef pos
    pos_des = control['pos'][(itr // itr_step) % 4]
    orn_des = control['orn'][(itr // itr_step) % 4]

    for i in range(num_envs):
        p0 = pos_cur[i].cpu().numpy()
        px = hammer_pos[i].cpu().numpy()
        gym.draw_env_rigid_contacts(viewer, envs[i], gymapi.Vec3(0.7, 0.2, 0.15), 0.3, True)
        # gym.add_lines(viewer, envs[i], 4, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0, 1, 0.1])

    # Solve for control (Operational Space Control)
    m_inv = torch.inverse(mm)
    m_eef = torch.inverse(j_eef @ m_inv @ torch.transpose(j_eef, 1, 2))
    orn_cur /= torch.norm(orn_cur, dim=-1).unsqueeze(-1)
    orn_err = orientation_error(orn_des, orn_cur)

    pos_err = kp * (pos_des - pos_cur)

    if not args.pos_control:
        pos_err *= 0

    dpose = torch.cat([pos_err, orn_err], -1)

    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (kp * dpose).unsqueeze(-1) - kv * mm @ dof_vel

    # update position targets
    pos_target = dof_pos + u

    # test gripper open
    grip_acts = control['close'][(itr // itr_step) % 4]
    pos_target[:, 7:9] = grip_acts.unsqueeze(-1)

    # set new position targets
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_target))

    # Set tensor action
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(u))

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

    down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)
    dist_reward_scale = 1.5
    rot_reward_scale = 0.5
    around_handle_reward_scale = 1.0
    open_reward_scale = 4.0
    finger_dist_reward_scale = 10.0
    action_penalty_scale = 0
    distX_offset = 0.04
    max_episode_length = 500

    # compute the reward after this state
    reward, above_hammer_reward, gripper_to_hammer_reward, gripper_downward_reward, gripper_hammer_y_reward, hammer_lift_reward = \
        compute_franka_reward(
        u, device,
        hammer_pose, None, None,
        franka_lfinger_pos, franka_lfinger_rot, franka_rfinger_pos, franka_rfinger_rot, down_dir,
        num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
        finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
    )

    print(itr)
    rewards.append(float(reward.squeeze().cpu().numpy()))
    gripper_hammer_dir_rewards.append(above_hammer_reward.squeeze().cpu().numpy())
    gripper_hammer_dis_rewards.append(gripper_to_hammer_reward.squeeze().cpu().numpy())
    gripper_down_rewards.append(gripper_downward_reward.squeeze().cpu().numpy())
    gripper_hammer_y_rewards.append(gripper_hammer_y_reward.squeeze().cpu().numpy())
    hammer_lift_rewards.append(hammer_lift_reward.squeeze().cpu().numpy())

    if itr >= 800:
        break

print(rewards)
plt.plot(range(799), rewards[1:])
plt.xlabel('Iteration')
plt.ylabel('Mean Reward')
plt.savefig("reward_plot.png")
plt.clf()

plt.plot(range(799), gripper_hammer_dir_rewards[1:])
plt.xlabel('Iteration')
plt.ylabel('Gripper Hammer dir Reward')
plt.savefig("gripper_hammer_dir_plot.png")
plt.clf()

plt.plot(range(799), gripper_hammer_dis_rewards[1:])
plt.xlabel('Iteration')
plt.ylabel('Gripper Hammer dis Reward')
plt.savefig("gripper_hammer_dis_plot.png")
plt.clf()

plt.plot(range(799), gripper_down_rewards[1:])
plt.xlabel('Iteration')
plt.ylabel('Gripper Downward Reward')
plt.savefig("gripper_down_plot.png")
plt.clf()

plt.plot(range(799), gripper_hammer_y_rewards[1:])
plt.xlabel('Iteration')
plt.ylabel('Gripper Y Reward')
plt.savefig("gripper_y_plot.png")
plt.clf()

plt.plot(range(799), hammer_lift_rewards[1:])
plt.xlabel('Iteration')
plt.ylabel('Hammer lift Reward')
plt.savefig("hammer_lift_plot.png")
plt.clf()

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)



