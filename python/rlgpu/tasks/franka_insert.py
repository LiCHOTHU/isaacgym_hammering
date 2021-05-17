# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os
import torch
import math

from rlgpu.utils.torch_jit_utils import *
from rlgpu.tasks.base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


class FrankaInsert(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, graphics_device, device):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.graphics_device = graphics_device
        self.device = device

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.noise_ = self.cfg["env"]["startRotationNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.reward_scale = self.cfg["env"]["rewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.pushDist = self.cfg["env"]["pushDist"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.



        # prop dimensions
        self.prop_width = 0.08
        self.prop_height = 0.08
        self.prop_length = 0.08
        self.prop_spacing = 0.09

        num_obs = 25
        num_acts = 9

        super().__init__(
            num_obs=num_obs,
            num_acts=num_acts,
            num_envs=self.cfg["env"]["numEnvs"],
            graphics_device=graphics_device,
            device=device
        )

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        # this default pos is pre-calculated for grasping
        self.franka_default_dof_pos = to_torch([0.7313, -0.3566, -0.5933, -1.9525, -0.1716, 1.7227, -1.9772, 0.0279, 0.0292],device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]

        self.global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).view(self.num_envs,
                                                                                                          -1)

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        # set downward direction vector
        self.down_dir = torch.Tensor([0, 0, -1]).to(self.device).view(1, 3)

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        print("actor_root_state_tensor shape: ", self.root_state_tensor.shape)
        print("rigid_body_tensor shape: ", self.rigid_body_states.shape)
        print("dof_state_tensor shape: ", self.dof_state.shape)
        print("root_state_tensor shape: ", self.root_state_tensor.shape)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.reset(torch.arange(self.num_envs, device=self.device))

        print("--------------------------Finish Initialize")

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = self.gym.create_sim(
            0, self.graphics_device, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../../assets"
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        # asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # get link index of panda hand, which we will use as end effector
        franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        self.franka_hand_index = franka_link_dict["panda_hand"]

        franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float,
                                        device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # create table asset
        table_dims = gymapi.Vec3(1.4, 1.4, 0.1)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        # create a box to insert
        box_dims = gymapi.Vec3(0.06, 0.06, 0.12)
        asset_options = gymapi.AssetOptions()
        # asset_options.density = 2000
        asset_options.fix_base_link = False
        box_asset = self.gym.create_box(self.sim, box_dims.x, box_dims.y, box_dims.z, asset_options)

        # create a hole for insertion, composed of front, back, left, right
        box_front_dims = gymapi.Vec3(0.04, 0.09, 0.04)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        box_front_asset = self.gym.create_box(self.sim, box_front_dims.x, box_front_dims.y, box_front_dims.z,
                                               asset_options)

        box_back_dims = gymapi.Vec3(0.04, 0.09, 0.04)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        box_back_asset = self.gym.create_box(self.sim, box_back_dims.x, box_back_dims.y, box_back_dims.z,
                                            asset_options)

        box_left_dims = gymapi.Vec3(0.17, 0.04, 0.04)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        box_left_asset = self.gym.create_box(self.sim, box_left_dims.x, box_left_dims.y, box_left_dims.z,
                                             asset_options)

        box_right_dims = gymapi.Vec3(0.17, 0.04, 0.04)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        box_right_asset = self.gym.create_box(self.sim, box_right_dims.x, box_right_dims.y, box_right_dims.z,
                                              asset_options)


        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        self.num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        self.num_table_dofs = self.gym.get_asset_dof_count(table_asset)
        self.num_box_bodies = self.gym.get_asset_rigid_body_count(box_asset)
        self.num_box_dofs = self.gym.get_asset_dof_count(box_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)
        print("num table bodies: ", self.num_table_bodies)
        print("num table dofs: ", self.num_table_dofs)
        print("num box bodies: ", self.num_box_bodies)
        print("num box dofs: ", self.num_box_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        franka_start_pose = gymapi.Transform()
        # Franka robot on the table
        franka_start_pose.p = gymapi.Vec3(0.0, 0.0, table_dims.z)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(-0.5, 0.0, 0.5 * table_dims.z)

        box_pose = gymapi.Transform()

        box_front_pose = gymapi.Transform()
        box_back_pose = gymapi.Transform()
        box_left_pose = gymapi.Transform()
        box_right_pose = gymapi.Transform()

        self.frankas = []
        self.boxes = []

        self.hand_idxs = []
        self.box_idxs = []
        self.franka_actor_idxs = []
        self.box_actor_idxs = []

        self.init_pos = []
        self.init_rot = []

        self.box_init_state = []

        self.default_prop_states = []
        self.prop_start = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            franka_actor_idx = self.gym.get_actor_index(env_ptr, franka_actor, gymapi.DOMAIN_SIM)
            self.franka_actor_idxs.append(franka_actor_idx)

            hand_idx = self.gym.find_actor_rigid_body_index(env_ptr, franka_actor, "panda_hand", gymapi.DOMAIN_SIM)
            self.hand_idxs.append(hand_idx)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
            color = gymapi.Vec3(0.98, 0.829, 0.592)
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # add box
            box_pose.p = gymapi.Vec3(-0.4841, -0.0137, 0.5612)
            box_pose.r = gymapi.Quat(-0.0048, -0.0734, -0.1039, 0.9919)
            box_actor = self.gym.create_actor(env_ptr, box_asset, box_pose, "box", i, 0)

            # set the goal position here
            self.goal_pos = gymapi.Vec3(-0.5, 0.0, table_dims.z + 0.5 * box_front_dims.z)
            self.sub_goal_pos = gymapi.Vec3(-0.5, 0.0, table_dims.z + box_front_dims.z + 0.5 * box_dims.z)

            self.goal = to_torch([self.goal_pos.x, self.goal_pos.y, self.goal_pos.z]).repeat(self.num_envs, 1)
            self.sub_goal = to_torch([self.sub_goal_pos.x, self.sub_goal_pos.y, self.sub_goal_pos.z]).repeat(self.num_envs, 1)

            # set the color of the box
            color = gymapi.Vec3(1, 0, 0)
            self.gym.set_rigid_body_color(env_ptr, box_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # get the global index of box in rigid body state tensor
            box_idx = self.gym.get_actor_rigid_body_index(env_ptr, box_actor, 0, gymapi.DOMAIN_SIM)
            self.box_idxs.append(box_idx)

            # get box actor index
            box_actor_idx = self.gym.get_actor_index(env_ptr, box_actor, gymapi.DOMAIN_SIM)
            self.box_actor_idxs.append(box_actor_idx)

            # get initial hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_hand")
            hand_pose = self.gym.get_rigid_transform(env_ptr, hand_handle)
            self.init_pos.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.init_rot.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

            # set box initial state
            self.box_init_state.append([box_pose.p.x, box_pose.p.y, box_pose.p.z,
                                        box_pose.r.x, box_pose.r.y, box_pose.r.z, box_pose.r.w,
                                        0, 0, 0, 0, 0, 0])


            # build a hole for the insertion
            # add box to hold the peg
            color = gymapi.Vec3(0.3, 0.2, 0.5)  # box color

            # add box front
            box_front_pose.p = gymapi.Vec3(self.goal_pos.x - 0.045 - 0.02, self.goal_pos.y, self.goal_pos.z)
            box_front_pose.r = gymapi.Quat(0, 0, 0)
            box_front_handle = self.gym.create_actor(env_ptr, box_front_asset, box_front_pose, "box_front", i, 0)

            # add box back
            box_back_pose.p = gymapi.Vec3(self.goal_pos.x + 0.045 + 0.02, self.goal_pos.y, self.goal_pos.z)
            box_back_pose.r = gymapi.Quat(0, 0, 0)
            box_back_handle = self.gym.create_actor(env_ptr, box_back_asset, box_back_pose, "box_back", i, 0)

            # add box left
            box_left_pose.p = gymapi.Vec3(self.goal_pos.x, self.goal_pos.y - 0.045 - 0.02, self.goal_pos.z)
            box_left_pose.r = gymapi.Quat(0, 0, 0)
            box_left_handle = self.gym.create_actor(env_ptr, box_left_asset, box_left_pose, "box_left", i, 0)

            # add box right
            box_right_pose.p = gymapi.Vec3(self.goal_pos.x, self.goal_pos.y + 0.045 + 0.02, self.goal_pos.z)
            box_right_pose.r = gymapi.Quat(0, 0, 0)
            box_right_handle = self.gym.create_actor(env_ptr, box_right_asset, box_right_pose, "box_right", i, 0)


            self.gym.set_rigid_body_color(env_ptr, box_front_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          color)  # box color
            self.gym.set_rigid_body_color(env_ptr, box_back_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          color)  # box color
            self.gym.set_rigid_body_color(env_ptr, box_left_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          color)  # box color
            self.gym.set_rigid_body_color(env_ptr, box_right_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          color)  # box color


            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.boxes.append(box_actor)


        self.box_init_state = to_torch(self.box_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13)

        self.box_idxs = to_torch(self.box_idxs, dtype=torch.long, device=self.device)
        self.hand_idxs = to_torch(self.hand_idxs, dtype=torch.long, device=self.device)

        self.franka_actor_idxs = to_torch(self.franka_actor_idxs, dtype=torch.long, device=self.device)
        self.box_actor_idxs = to_torch(self.box_actor_idxs, dtype=torch.long, device=self.device)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_hand")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger")

        # initial hand position and orientation tensors
        self.init_pos = torch.Tensor(self.init_pos).view(num_envs, 3).to(self.device)
        self.init_rot = torch.Tensor(self.init_rot).view(num_envs, 4).to(self.device)



    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.box_pose, self.goal, self.sub_goal,
            self.franka_lfinger_pos, self.franka_lfinger_rot, self.franka_rfinger_pos, self.franka_rfinger_rot,
            self.hand_pos, self.hand_rot,
            self.down_dir,
            self.num_envs,
            self.reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        # configuration space control
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]

        self.box_pose = self.root_state_tensor[self.box_actor_idxs, 0:7]
        self.box_pos = self.root_state_tensor[self.box_actor_idxs, 0:3]
        self.box_rot = self.root_state_tensor[self.box_actor_idxs, 3:7]
        self.box_linvel = self.root_state_tensor[self.box_actor_idxs, 7:10]
        self.box_angvel = self.root_state_tensor[self.box_actor_idxs, 10:13]

        # EEF control
        self.franka_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.franka_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.franka_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.franka_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        self.hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        self.hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]

        dof_pos_scaled = (2.0 * (self.franka_dof_pos - self.franka_dof_lower_limits)
                          / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)

        # TODO object observation compute the cornor
        # TODO current obs: (7dof+2gripper)*2 (pos+vel)(franka) + (3pos+4quat)(box) = 25
        self.obs_buf = torch.cat((dof_pos_scaled, self.franka_dof_vel * self.dof_vel_scale, self.box_pose), dim=-1)

        return self.obs_buf

    def reset(self, env_ids):
        # reset box
        self.root_state_tensor[self.box_actor_idxs[env_ids]] = self.box_init_state[env_ids].clone()
        object_indices = torch.unique(self.box_actor_idxs[env_ids]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset franka
        pos = self.franka_default_dof_pos
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        # reset franka
        franka_indices = self.franka_actor_idxs[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(franka_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(franka_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        self.box_pos = self.rigid_body_states.view(-1, 13)[self.box_idxs, :3]
        self.hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        # TODO setting actions
        self.actions[:, -2:] = 0
        targets = self.franka_dof_targets[:,
                  :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))

    def quat_axis(self, q, axis=0):
        basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
        basis_vec[:, axis] = 1
        return quat_rotate(q, basis_vec)

    def orientation_error(self, desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        '''
        # debug
        for i in range(self.num_envs):
            p0 = self.hammer_pos[i].cpu().numpy()
            p1 = ((self.franka_lfinger_pos[i]+self.franka_rfinger_pos[i])/2.0).cpu().numpy()
            self.gym.add_lines(self.viewer, self.envs[i], 9, [p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]], [1, 0, 0])  # red
        '''

        # debug viz
        self.debug_viz = False
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            '''
            for i in range(self.num_envs):

                px = (self.hammer_pos[i] + quat_apply(self.hammer_rot[i],
                                                      to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.hammer_pos[i] + quat_apply(self.hammer_rot[i],
                                                      to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.hammer_pos[i] + quat_apply(self.hammer_rot[i],
                                                      to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.hammer_pos[i].cpu().numpy()

                p_new = px + py + pz

                self.gym.add_lines(self.viewer, self.envs[i], 9, [p0[0], p0[1], p0[2], p_new[0], p_new[1], p_new[2]],
                                   [0.85, 0.1, 0.1])

                px = (self.peg_pos[i] + quat_apply(self.peg_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.peg_pos[i] + quat_apply(self.peg_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.peg_pos[i] + quat_apply(self.peg_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.peg_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])


                px = (self.hammer_pos[i] + quat_apply(self.hammer_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.hammer_pos[i] + quat_apply(self.hammer_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.hammer_pos[i] + quat_apply(self.hammer_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.hammer_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 4, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])
                '''


#####################################################################
###=========================jit functions=========================###
#####################################################################


def compute_franka_reward(
        reset_buf, progress_buf, actions,
        box_pose, goal, sub_goal,
        franka_lfinger_pos, franka_lfinger_rot, franka_rfinger_pos, franka_rfinger_rot,
        hand_pos, hand_rot,
        down_dir,
        num_envs,
        reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # set some init data for calculation
    box_pos = box_pose[:, :3]
    franka_gripper_pos = (franka_rfinger_pos + franka_lfinger_pos) * 0.5

    drop_offset = 0.09
    box_gripper_dist = torch.norm(franka_gripper_pos - box_pos, dim=-1)

    # is_dropped = torch.where(box_gripper_dist > drop_offset, torch.tanh(10 * (box_gripper_dist - drop_offset)), to_torch([0.0]).repeat(num_envs))
    is_dropped = torch.where(box_gripper_dist > drop_offset, to_torch([1.0]).repeat(num_envs),
                             to_torch([0.0]).repeat(num_envs))

    drop_penaly = -10
    rewards = drop_penaly * is_dropped

    # reaching to sub-goal
    box_sub_goal_dist = torch.norm(sub_goal - box_pos, dim=-1)
    box_sub_goal_weight = 3
    box_sub_goal_reward = 1 - torch.tanh(10.0 * box_sub_goal_dist)
    # print(box_sub_goal_reward)
    rewards += box_sub_goal_reward * box_sub_goal_weight

    is_closed_enough = torch.where(box_sub_goal_dist < 0.02, to_torch([1.0]).repeat(num_envs), to_torch([0.0]).repeat(num_envs))

    # reaching to goal
    box_goal_weight = 5
    box_goal_dist = torch.norm(goal - box_pos, dim=-1)
    box_goal_reward = (1 - torch.tanh(10.0 * box_goal_dist))* is_closed_enough
    # print(box_goal_reward)
    rewards += box_goal_reward * box_goal_weight



    '''
    # reaching reward
    gripper_pos = (franka_lfinger_pos + franka_rfinger_pos) / 2.0
    reach_dist = torch.norm(gripper_pos - box_pos, dim=-1).unsqueeze(-1)
    zero = to_torch([0]).repeat(num_envs, 1)
    reach_diff = torch.max(zero, reach_dist)

    reach_reward = 1 - torch.tanh(10.0 * reach_diff)
    rewards = reach_reward.squeeze()

    box_pos_xy = box_pos[:, 0:2]
    goal_dist = torch.norm(goal - box_pos_xy, dim=-1).unsqueeze(-1)

    box_goal_reward = 1 - torch.tanh(10.0 * goal_dist)

    rewards += box_goal_reward.squeeze()
    '''
    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    rewards -= action_penalty_scale * action_penalty

    reset_buf = torch.where(franka_lfinger_pos[:, 0] < box_pose[:, 0] - distX_offset,
                            torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(franka_rfinger_pos[:, 0] < box_pose[:, 0] - distX_offset,
                            torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)



    return rewards, reset_buf


@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos,
                             drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos)
    global_drawer_rot, global_drawer_pos = tf_combine(
        drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos)

    return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos