# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import operator
from copy import deepcopy
import random

from isaacgym import gymapi
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, apply_random_samples, check_buckets

import torch


# Base class for RL tasks
class BaseTask():

    def __init__(self, num_obs, num_acts, num_envs, graphics_device, device):
        self.gym = gymapi.acquire_gym()
        self.device = device

        self.num_obs = num_obs
        self.num_actions = num_acts
        self.num_envs = num_envs

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=device, dtype=torch.long)
        self.extras = []

        self.original_props = {}
        self.first_randomization = True

        self.enable_viewer_sync = True

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.viewer = None

        self.obs_noise = None
        self.action_noise = None
        self.last_step = -1
        self.last_rand_step = -1

        # if running with a viewer, set up keyboard shortcuts and camera
        if graphics_device != -1:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 7.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 7.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)
            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    # set gravity based on up axis and return axis index
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def step(self, actions):
        if self.action_noise is not None:
            actions = self.action_noise(actions)
        # apply actions
        self.pre_physics_step(actions)

        # step physics
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        if self.obs_noise is not None:
            self.obs_buf = self.obs_noise(self.obs_buf)

    def render(self, sync_frame_time=False):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)

    # Apply randomizations only on resets, due to current PhysX limitations
    def apply_randomizations(self, dr_params):
        # If we don't have a randomization frequency, randomize every time
        rand_freq = 1
        if "frequency" in dr_params:
            rand_freq = dr_params["frequency"]

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = range(self.num_envs)
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf), torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False)
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1 / sched_step * \
                        min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1 * \
                            (1 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor): return op(
                        tensor, torch.randn_like(tensor) * var + mu)

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1 * (1 - sched_scaling)
                        hi = hi * sched_scaling + 1 * (1 - sched_scaling)

                    def noise_lambda(tensor): return op(
                        tensor, torch.rand_like(tensor) * (hi - lo) + lo)

                if nonphysical_param == 'actions':
                    self.action_noise = noise_lambda
                else:
                    self.obs_noise = noise_lambda

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)}
            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)
            self.gym.set_sim_params(self.sim, prop)

        for actor, actor_properties in dr_params["actor_params"].items():
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)

                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue

                    prop = param_getters_map[prop_name](env, handle)

                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                apply_random_samples(
                                    p, og_p, attr, attr_randomization_params, self.last_step)
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            apply_random_samples(
                                prop, self.original_props[prop_name], attr, attr_randomization_params, self.last_step)

                    param_setters_map[prop_name](env, handle, prop)

        self.first_randomization = False

    def create_sim(self):
        raise NotImplementedError

    def pre_physics_step(self, actions):
        raise NotImplementedError

    def post_physics_step(self):
        raise NotImplementedError
