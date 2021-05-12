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


class FrankaHammer(BaseTask):

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
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1/60.

        # prop dimensions
        self.prop_width = 0.08
        self.prop_height = 0.08
        self.prop_length = 0.08
        self.prop_spacing = 0.09

        num_obs = 25
        num_acts = 9

        # TODO set hammer being grasped
        self.set_grasp = True

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
        self.franka_default_dof_pos = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]

        self.global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

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

        # TODO for debugging
        # self.init_grasp()


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
        hammer_asset_file = "urdf/hammer_convex.urdf"
        peg_asset_file = "urdf/peg.urdf"


        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        # asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # get link index of panda hand, which we will use as end effector
        franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        self.franka_hand_index = franka_link_dict["panda_hand"]

        franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # create table asset
        table_dims = gymapi.Vec3(0.8, 1.2, 0.4)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        # create hammer asset
        asset_options = gymapi.AssetOptions()
        asset_options.density = 2000
        hammer_asset = self.gym.load_asset(self.sim, asset_root, hammer_asset_file, asset_options)

        # create a box for peg, composed of bottom, right, left and top part
        box_bottom_dims = gymapi.Vec3(0.16, 0.16, 0.1)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        box_bottom_asset = self.gym.create_box(self.sim, box_bottom_dims.x, box_bottom_dims.y, box_bottom_dims.z,
                                               asset_options)

        box_top_dims = gymapi.Vec3(0.16, 0.16, 0.04)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        box_top_asset = self.gym.create_box(self.sim, box_top_dims.x, box_top_dims.y, box_top_dims.z,
                                               asset_options)

        box_left_dims = gymapi.Vec3(0.06, 0.16, 0.04)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        box_left_asset = self.gym.create_box(self.sim, box_left_dims.x, box_left_dims.y, box_left_dims.z,
                                               asset_options)

        box_right_dims = gymapi.Vec3(0.06, 0.16, 0.04)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        box_right_asset = self.gym.create_box(self.sim, box_right_dims.x, box_right_dims.y, box_right_dims.z,
                                               asset_options)


        # create peg asset
        asset_options = gymapi.AssetOptions()
        asset_options.density = 2000
        peg_asset = self.gym.load_asset(self.sim, asset_root, peg_asset_file, asset_options)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        self.num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        self.num_table_dofs = self.gym.get_asset_dof_count(table_asset)
        self.num_hammer_bodies = self.gym.get_asset_rigid_body_count(hammer_asset)
        self.num_hammer_dofs = self.gym.get_asset_dof_count(hammer_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)
        print("num table bodies: ", self.num_table_bodies)
        print("num table dofs: ", self.num_table_dofs)
        print("num hammer bodies: ", self.num_hammer_bodies)
        print("num hammer dofs: ", self.num_hammer_dofs)

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
        franka_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(-0.5, 0.0, 0.5 * table_dims.z)

        hammer_pose = gymapi.Transform()
        box_bottom_pose = gymapi.Transform()
        box_top_pose = gymapi.Transform()
        box_left_pose = gymapi.Transform()
        box_right_pose = gymapi.Transform()
        peg_pose = gymapi.Transform()

        self.frankas = []
        self.hammers = []
        self.cameras = []

        self.hand_idxs = []
        self.hammer_idxs = []
        self.hand_idxs = []
        self.peg_idxs = []
        self.franka_actor_idxs = []
        self.hammer_actor_idxs = []
        self.peg_actor_idxs = []

        self.init_pos = []
        self.init_rot = []

        self.hammer_init_state = []
        self.peg_init_state = []
        self.camera_tensors = []


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

            # add hammer
            # avoid random position
            hammer_pose.p.x = table_pose.p.x + 0.
            hammer_pose.p.y = table_pose.p.y - 0.1
            # hammer_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
            # hammer_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
            hammer_pose.p.z = table_dims.z + 0.03

            if self.set_grasp:
                hammer_pose.p = gymapi.Vec3(-0.4606, -0.0269, 0.6031)
                hammer_pose.r = gymapi.Quat(-0.0155,  0.0465, -0.6376,  0.7688)

            # hammer_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            # avoid random rotation
            hammer_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -math.pi/2)

            hammer_actor = self.gym.create_actor(env_ptr, hammer_asset, hammer_pose, "hammer", i, 0)
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env_ptr, hammer_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # get the global index of hammer in rigid body state tensor
            hammer_idx = self.gym.get_actor_rigid_body_index(env_ptr, hammer_actor, 0, gymapi.DOMAIN_SIM)
            self.hammer_idxs.append(hammer_idx)

            # get hammer actor index
            hammer_actor_idx = self.gym.get_actor_index(env_ptr, hammer_actor, gymapi.DOMAIN_SIM)
            self.hammer_actor_idxs.append(hammer_actor_idx)

            # get initial hand pose
            hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_hand")
            hand_pose = self.gym.get_rigid_transform(env_ptr, hand_handle)
            self.init_pos.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
            self.init_rot.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])


            # set hammer initial state
            self.hammer_init_state.append([hammer_pose.p.x, hammer_pose.p.y, hammer_pose.p.z,
                                           hammer_pose.r.x, hammer_pose.r.y, hammer_pose.r.z,hammer_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            # add box to hold the peg
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)) # box color

            # add box bottom
            box_relative_pose = gymapi.Transform()
            box_relative_pose.p = gymapi.Vec3(-0.25, 0.25, 0)
            box_bottom_pose.p.x = table_pose.p.x + box_relative_pose.p.x
            box_bottom_pose.p.y = table_pose.p.y + box_relative_pose.p.y
            box_bottom_pose.p.z = table_dims.z + 0.5 * box_bottom_dims.z
            box_bottom_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi/2)
            box_bottom_handle = self.gym.create_actor(env_ptr, box_bottom_asset, box_bottom_pose, "box_bottom", i, 0)

            # add box left
            box_hole_length = box_bottom_dims.x - box_left_dims.x - box_right_dims.x
            box_left_pose.p.x = box_bottom_pose.p.x
            box_left_pose.p.y = box_bottom_pose.p.y - 0.5 * box_hole_length - 0.5 * box_left_dims.x
            box_left_pose.p.z = table_dims.z + box_bottom_dims.z + 0.5 * box_left_dims.z
            box_left_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi/2)
            box_left_handle = self.gym.create_actor(env_ptr, box_left_asset, box_left_pose, "box_left", i, 0)

            # add box right
            box_right_pose.p.x = box_bottom_pose.p.x
            box_right_pose.p.y = box_bottom_pose.p.y + 0.5 * box_hole_length + 0.5 * box_left_dims.x
            box_right_pose.p.z = box_left_pose.p.z
            box_right_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi/2)
            box_right_handle = self.gym.create_actor(env_ptr, box_right_asset, box_right_pose, "box_right", i, 0)

            # add box top
            box_top_pose.p.x = box_bottom_pose.p.x
            box_top_pose.p.y = box_bottom_pose.p.y
            box_top_pose.p.z = box_bottom_pose.p.z + 0.5 * box_bottom_dims.z + box_left_dims.x
            box_top_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi/2)
            box_top_handle = self.gym.create_actor(env_ptr, box_top_asset, box_top_pose, "box_top", i, 0)

            self.gym.set_rigid_body_color(env_ptr, box_bottom_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color) # box color
            self.gym.set_rigid_body_color(env_ptr, box_top_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          color)  # box color
            self.gym.set_rigid_body_color(env_ptr, box_left_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          color)  # box color
            self.gym.set_rigid_body_color(env_ptr, box_right_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          color)  # box color

            # add peg
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))  # box color
            peg_pose.p.x = box_bottom_pose.p.x + 0.16 # set the initial peg position
            peg_pose.p.y = box_bottom_pose.p.y
            peg_pose.p.z = box_left_pose.p.z
            peg_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), math.pi / 2)
            peg_actor = self.gym.create_actor(env_ptr, peg_asset, peg_pose, "peg", i, 0)
            self.gym.set_rigid_body_color(env_ptr, peg_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color) # peg color

            # get the global index of peg in rigid body state tensor
            peg_idx = self.gym.get_actor_rigid_body_index(env_ptr, peg_actor, 0, gymapi.DOMAIN_SIM)
            self.peg_idxs.append(peg_idx)

            # get peg actor index
            peg_actor_idx = self.gym.get_actor_index(env_ptr, peg_actor, gymapi.DOMAIN_SIM)
            self.peg_actor_idxs.append(peg_actor_idx)

            self.peg_init_state.append([peg_pose.p.x, peg_pose.p.y, peg_pose.p.z,
                                               peg_pose.r.x, peg_pose.r.y, peg_pose.r.z, peg_pose.r.w,
                                               0, 0, 0, 0, 0, 0])

            '''
            # add camera
            camera_props = gymapi.CameraProperties()
            camera_props.horizontal_fov = 75.0
            camera_props.width = 128
            camera_props.height = 128

            camera_props.enable_tensors = True
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)

            transform = gymapi.Transform()
            transform.p = hammer_pose.p
            transform.p.z += 0.5 # set the camera to be above the hammer
            transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(180.0))

            self.gym.set_camera_transform(camera_handle, env_ptr, transform)

            # obtain camera tensor
            camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)

            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)

            self.camera_tensors.append(torch_camera_tensor)

            # self.gym.write_camera_image_to_file(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR, "test_image.png")
            # self.cameras.append(camera_handle)
            '''

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.hammers.append(hammer_actor)



        self.hammer_init_state = to_torch(self.hammer_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.peg_init_state = to_torch(self.peg_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)

        self.hammer_idxs = to_torch(self.hammer_idxs, dtype=torch.long, device=self.device)
        self.hand_idxs = to_torch(self.hand_idxs, dtype=torch.long, device=self.device)
        self.peg_idxs = to_torch(self.peg_idxs, dtype=torch.long, device=self.device)

        self.franka_actor_idxs = to_torch(self.franka_actor_idxs, dtype=torch.long, device=self.device)
        self.hammer_actor_idxs = to_torch(self.hammer_actor_idxs, dtype=torch.long, device=self.device)
        self.peg_actor_idxs = to_torch(self.peg_actor_idxs, dtype=torch.long, device=self.device)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_hand")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger")

        # initial hand position and orientation tensors
        self.init_pos = torch.Tensor(self.init_pos).view(num_envs, 3).to(self.device)
        self.init_rot = torch.Tensor(self.init_rot).view(num_envs, 4).to(self.device)

        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)

        self.init_data()

    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_hand")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_leftfinger")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_rightfinger")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        finger_pose.r = lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        franka_local_grasp_pose = hand_pose_inv * finger_pose
        franka_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))
        self.franka_local_grasp_pos = to_torch([franka_local_grasp_pose.p.x, franka_local_grasp_pose.p.y,
                                                franka_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.franka_local_grasp_rot = to_torch([franka_local_grasp_pose.r.x, franka_local_grasp_pose.r.y,
                                                franka_local_grasp_pose.r.z, franka_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.franka_grasp_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_grasp_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_grasp_rot[..., -1] = 1  # xyzw
        self.franka_lfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_rfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_lfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_rfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.hammer_pose, self.peg_pose, self.peg_linvel, self.peg_init_state,
            self.franka_lfinger_pos, self.franka_lfinger_rot, self.franka_rfinger_pos, self.franka_rfinger_rot, self.down_dir,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        self.hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        self.hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]

        self.hammer_pose = self.root_state_tensor[self.hammer_actor_idxs, 0:7]
        self.hammer_pos = self.root_state_tensor[self.hammer_actor_idxs, 0:3]
        self.hammer_rot = self.root_state_tensor[self.hammer_actor_idxs, 3:7]
        self.hammer_linvel = self.root_state_tensor[self.hammer_actor_idxs, 7:10]
        self.hammer_angvel = self.root_state_tensor[self.hammer_actor_idxs, 10:13]

        self.peg_pose = self.root_state_tensor[self.peg_actor_idxs, 0:7]
        self.peg_pos = self.root_state_tensor[self.peg_actor_idxs, 0:3]
        self.peg_rot = self.root_state_tensor[self.peg_actor_idxs, 3:7]
        self.peg_linvel = self.root_state_tensor[self.peg_actor_idxs, 7:10]
        self.peg_angvel = self.root_state_tensor[self.peg_actor_idxs, 10:13]

        self.franka_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.franka_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.franka_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.franka_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]


        dof_pos_scaled = (2.0 * (self.franka_dof_pos - self.franka_dof_lower_limits)
                          / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)

        self.obs_buf = torch.cat((dof_pos_scaled, self.franka_dof_vel * self.dof_vel_scale, self.hammer_pose), dim=-1)

        return self.obs_buf

    def reset(self, env_ids):
        # reset hammer
        self.root_state_tensor[self.hammer_actor_idxs[env_ids]] = self.hammer_init_state[env_ids].clone()

        # reset peg
        self.root_state_tensor[self.peg_actor_idxs[env_ids]] = self.peg_init_state[env_ids].clone()

        object_indices = torch.unique(torch.cat([self.hammer_actor_idxs[env_ids], self.peg_actor_idxs[env_ids]]).to(torch.int32))

        # object_indices = self.hammer_actor_idxs[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        if self.set_grasp:
            pos = to_torch([0.1397, -0.2689, -0.0908, -1.6391, 0.0319, 1.4131, -0.8710, 0.0148, 0.0154], device=self.device)

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

        self.hammer_pos = self.rigid_body_states.view(-1, 13)[self.hammer_idxs, :3]
        self.hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))

    # TODO using IK to solve the grasping
    def init_grasp(self):
        lifted = False

        # Create a tensor noting whether the hand should return to the initial position
        hand_restart = torch.full([self.num_envs], False, dtype=torch.bool).to(self.device)

        # hand orientation for grasping
        down_q = torch.stack(self.num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(self.device).view((self.num_envs, 4))

        # jacobian entries corresponding to franka hand
        j_eef = self.jacobian[:, self.franka_hand_index - 1, :]

        itr = 0

        while not lifted:
            print(itr)
            itr += 1
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # refresh tensors
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)

            self.hammer_pos = self.rigid_body_states.view(-1, 13)[self.hammer_idxs, :3]
            self.hammer_rot = self.rigid_body_states.view(-1, 13)[self.hammer_idxs, 3:7]

            self.hand_pos = self.rigid_body_states[:, self.hand_handle, 0:3]
            self.hand_rot = self.rigid_body_states[:, self.hand_handle, 3:7]

            to_hammer = self.hammer_pos - self.hand_pos
            hammer_dist = torch.norm(to_hammer, dim=-1).unsqueeze(-1)
            hammer_dir = to_hammer / hammer_dist
            down_dir = torch.Tensor([0, 0, -1]).to(self.device).view(1, 3)
            hammer_dot = hammer_dir @ down_dir.view(3, 1)

            # how far the hand should be from hammer for grasping
            grasp_offset = 0.1

            dof_pos = self.dof_state[:, 0].view(self.num_envs, 9, 1)
            # determine if we're holding the hammer (grippers are closed and hammer is near)
            gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]

            # TODO: hammer height might be different if we train with multi hammers
            hammer_height = 0.035
            gripped = (gripper_sep < hammer_height) & (hammer_dist < grasp_offset + 0.5 * hammer_height)

            # hammer corner coords, used to determine grasping yaw
            hammer_half_size = 0.5 * hammer_height
            corner_coord = torch.Tensor([hammer_half_size, hammer_half_size, hammer_half_size])
            corners = torch.stack(self.num_envs * [corner_coord]).to(self.device)


            yaw_q = self.cube_grasping_yaw(self.hammer_rot, corners)
            hammer_yaw_dir = self.quat_axis(yaw_q, 0)
            hand_yaw_dir = self.quat_axis(self.hand_rot, 0)
            yaw_dot = torch.bmm(hammer_yaw_dir.view(self.num_envs, 1, 3), hand_yaw_dir.view(self.num_envs, 3, 1)).squeeze(-1)

            # determine if we have reached the initial position; if so allow the hand to start moving to the hammer
            to_init = self.init_pos - self.hand_pos
            init_dist = torch.norm(to_init, dim=-1)
            hand_restart = (hand_restart & (init_dist > 0.02)).squeeze(-1)
            return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)

            # if hand is above hammer, descend to grasp offset
            # otherwise, seek a position above the hammer
            above_hammer = ((hammer_dot >= 0.99) & (yaw_dot >= 0.95) & (hammer_dist < grasp_offset * 3)).squeeze(-1)
            grasp_pos = self.hammer_pos.clone()
            grasp_pos[:, 2] = torch.where(above_hammer, self.hammer_pos[:, 2] + grasp_offset, self.hammer_pos[:, 2] + grasp_offset * 2.5)

            # compute goal position and orientation
            goal_pos = torch.where(return_to_start, self.init_pos, grasp_pos)
            goal_rot = torch.where(return_to_start, self.init_rot, quat_mul(down_q, quat_conjugate(yaw_q)))

            # compute position and orientation error
            pos_err = goal_pos - self.hand_pos
            orn_err = self.orientation_error(goal_rot, self.hand_rot)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

            # solve damped least squares
            j_eef_T = torch.transpose(j_eef, 1, 2)
            d = 0.05  # damping term
            lmbda = torch.eye(6).to(self.device) * (d ** 2)
            u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 9, 1)

            # update position targets
            pos_target = dof_pos + u

            # gripper actions depend on distance between hand and hammer
            close_gripper = (hammer_dist < grasp_offset + 0.01) | gripped

            # always open the gripper above a certain height, dropping the hammer and restarting from the beginning
            # hand_restart = hand_restart | (self.hammer_pos[:, 2] > 0.65)
            # keep_going = torch.logical_not(hand_restart)
            # close_gripper = close_gripper & keep_going.unsqueeze(-1)
            grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * self.num_envs).to(self.device),
                                    torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device))
            pos_target[:, 7:9] = grip_acts.unsqueeze(-1)


            # TODO if 95% of the envs lifted the hammer, the nwe can start training hammering task
            lifted_envs = torch.sum(((self.hammer_pos[:, 2] > 0.65) & (gripped.squeeze())).float()).cpu().numpy()
            if lifted_envs > self.num_envs * 0.95:
                lifted = True

            # set new position targets
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_target))

            # update viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
        print("finish initial grasping the hammer, valid to start training")


    def quat_axis(self, q, axis=0):
        basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
        basis_vec[:, axis] = 1
        return quat_rotate(q, basis_vec)


    def cube_grasping_yaw(self, q, corners):
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

        hammer_rot = self.hammer_rot

        x_axis = to_torch([1, 0, 0]).repeat(self.num_envs, 1)
        y_axis = to_torch([0, 1, 0]).repeat(self.num_envs, 1)
        z_axis = to_torch([0, 0, 1]).repeat(self.num_envs, 1)

        p0 = self.hammer_pos
        px = (self.hammer_pos + quat_apply(hammer_rot, x_axis)) - p0
        py = (self.hammer_pos + quat_apply(hammer_rot, y_axis)) - p0
        pz = (self.hammer_pos + quat_apply(hammer_rot, z_axis)) - p0

        x = -0.073
        y = -0.073
        z = 0.005

        strike_pos = p0 + x * px + y * py + z * pz

        # debug viz
        self.debug_viz = False
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                '''
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
                '''

                px = (strike_pos[i] + quat_apply(self.hammer_rot[i],
                                                   to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (strike_pos[i] + quat_apply(self.hammer_rot[i],
                                                   to_torch([0, -1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (strike_pos[i] + quat_apply(self.hammer_rot[i],
                                                   to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = strike_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 9, [p0[0], p0[1], p0[2], px[0], px[1], px[2]],
                                   [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 9, [p0[0], p0[1], p0[2], py[0], py[1], py[2]],
                                   [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 9, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
                                   [0.1, 0.1, 0.85])




                '''
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


# @torch.jit.script
def compute_franka_reward_grasp(
    reset_buf, progress_buf, actions,
    hammer_pose, peg_pose, peg_linvel, peg_init_state,
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

    to_hammer = hammer_pos - franka_gripper_pos
    gripper_hammer_dist = torch.norm(to_hammer, dim=-1).unsqueeze(-1)
    gripper_hammer_dir = to_hammer / gripper_hammer_dist

    gripper_hammer_dot = gripper_hammer_dir @ down_dir.view(3, 1)
    above_hammer_reward = (gripper_above_hammer_weight * (gripper_hammer_dot ** 2)*torch.sign(gripper_hammer_dot)).squeeze()


    # calculate the gripper to hammer distance
    d = torch.norm(franka_gripper_pos - hammer_pos, p=2, dim=-1)
    dist_reward = torch.where(d > grasp_offset, 1.0 - torch.tanh(d), to_torch([1.0]).repeat(num_envs))

    rewards = dist_reward * gripper_to_hammer_dist_weight
    rewards += above_hammer_reward

    # calculate the gripper towards direction should be facing downward
    gripper_rot = (franka_lfinger_rot + franka_rfinger_rot) / 2.0
    z_dir = to_torch([0, 0, 1]).repeat(num_envs, 1)
    gripper_z_dir = quat_apply(gripper_rot, z_dir)
    gripper_z_dot = gripper_z_dir @ down_dir.view(3, 1)
    gripper_downward_reward = (torch.sign(gripper_z_dot)*(gripper_z_dot**2) * gripper_downward_weight).squeeze()
    rewards += gripper_downward_reward

    # calculate the reward if the gripper_y and hammer_y alignment
    hammer_rot = hammer_pose[:, 3:7]
    y_dir = to_torch([0, 1, 0]).repeat(num_envs, 1)
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

    # determine if we're holding the hammer (grippers are closed and hammer is near)
    gripper_sep = torch.norm(franka_lfinger_pos - franka_rfinger_pos, dim =-1).unsqueeze(-1)
    gripped = (gripper_sep < gripper_close_offset) & (gripper_hammer_dist < grasp_offset)

    # add penalty for dropping off the table
    hammer_drop_table = (hammer_pos[:, 2] < 0.4).float().squeeze()
    hammer_drop_penalty = -1000 * hammer_drop_table
    rewards += hammer_drop_penalty

    # add reward for lifting the hammer
    hammer_lift_height = torch.clamp(hammer_pos[:, 2] - hammer_init_height, 0, hammer_max_height)
    hammer_lift_reward = gripped.float().squeeze() * hammer_lift_height * hammer_lift_weight
    rewards += hammer_lift_reward

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    rewards -= action_penalty_scale * action_penalty


    # TODO Compute the grasping reward
    reset_buf = torch.where(franka_lfinger_pos[:, 0] < hammer_pose[:, 0] - distX_offset,
                            torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(franka_rfinger_pos[:, 0] < hammer_pose[:, 0] - distX_offset,
                            torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf


def compute_franka_reward(
    reset_buf, progress_buf, actions,
    hammer_pose, peg_pose, peg_linvel, peg_init_state,
    franka_lfinger_pos, franka_lfinger_rot, franka_rfinger_pos, franka_rfinger_rot, down_dir,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor]


    # set some init data for calculation
    hammer_pos = hammer_pose[:, :3]
    peg_pos = peg_pose[:, :3]
    peg_init_pos = peg_init_state[:, :3]


    hammer_rot = hammer_pose[:, 3:7]

    x_axis = to_torch([1, 0, 0]).repeat(num_envs, 1)
    y_axis = to_torch([0, 1, 0]).repeat(num_envs, 1)
    z_axis = to_torch([0, 0, 1]).repeat(num_envs, 1)

    p0 = hammer_pos
    px = (hammer_pos + quat_apply(hammer_rot, x_axis)) - p0
    py = (hammer_pos + quat_apply(hammer_rot, y_axis)) - p0
    pz = (hammer_pos + quat_apply(hammer_rot, z_axis)) - p0

    x = -0.073
    y = -0.073
    z = 0.005

    strike_pos = p0 + x * px + y * py + z * pz
    strike_dir = (strike_pos + quat_apply(hammer_rot, -y_axis)) - strike_pos

    hammer_peg_weight = 1.0
    peg_dist_weight = 10.0
    hammer_dir_weight = 0.5
    hammer_drop_penalty_weight = 5.0

    # add penalty for dropping off the table or dropping from hand
    finger_pos = (franka_lfinger_pos + franka_rfinger_pos) / 2.0
    finger_hammer_dist = torch.norm(finger_pos - hammer_pos, dim=-1).unsqueeze(-1)

    hammer_drop_finger = (finger_hammer_dist > 0.06).squeeze()
    hammer_drop_table = (hammer_pos[:, 2] < 0.5).squeeze()

    hammer_drop = (hammer_drop_finger | hammer_drop_table).float()

    hammer_drop_penalty = -hammer_drop_penalty_weight * hammer_drop
    rewards = hammer_drop_penalty


    # add reward for distance between peg and hammer
    hammer_to_peg = strike_pos - peg_pos
    hammer_peg_dist = torch.norm(hammer_to_peg, dim=-1).unsqueeze(-1)

    hammer_peg_reward = ((1.0 - torch.tanh(hammer_peg_dist)) * hammer_peg_weight).squeeze()
    rewards += hammer_peg_reward

    # add reward for strike direction alignment
    strike_dir = (strike_pos + quat_apply(hammer_rot, -y_axis)) - strike_pos
    strike_x_dot = torch.bmm(strike_dir.view(num_envs, 1, 3), -x_axis.view(num_envs, 3, 1)).squeeze()

    rewards += strike_x_dot * hammer_dir_weight

    # add reward for distance peg goes in
    hammer_close_peg = (hammer_peg_dist < 0.05).float().squeeze()
    peg_dist = torch.norm(peg_pos - peg_init_pos, dim=-1).squeeze()
    peg_dist_reward = torch.tanh(peg_dist) * hammer_close_peg * peg_dist_weight

    rewards += peg_dist_reward

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)
    rewards -= action_penalty_scale * action_penalty

    # TODO Compute the grasping reward
    reset_buf = torch.where(franka_lfinger_pos[:, 0] < hammer_pose[:, 0] - distX_offset,
                            torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(franka_rfinger_pos[:, 0] < hammer_pose[:, 0] - distX_offset,
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