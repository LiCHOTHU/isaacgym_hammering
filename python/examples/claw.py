"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Claw picking example
--------------------
Each claw environment repeatedly reaches down and grabs a block
- Illustrates using an MJCF to load a robot model
- Scripting sequences of actions with multiple environments
- Contact normal and force rendering along with user customization of how they are drawn
"""

import queue
import math
import random
import time
import numpy as np

from isaacgym import gymapi
from isaacgym import gymutil


class CmdArmPos:
    def __init__(self, x_target, z_target):
        self.x_target = x_target
        self.z_target = z_target


class CmdGripperRot:
    def __init__(self, angle_target):
        self.angle_target = angle_target


class CmdGripperHeight:
    def __init__(self, y_target):
        self.y_target = y_target


class CmdGripperClose:
    def __init__(self, timeout):
        self.timeout = timeout


class CmdGripperOpen:
    def __init__(self):
        pass


class CmdSleep:
    def __init__(self, duration):
        self.duration = duration


class Claw:
    # Claw contains 7 degrees of freedom
    NUM_DOFS = 7
    # Index representing each degree of freedom
    DOF_ARM_X = 0
    DOF_ARM_Z = 1
    DOF_ARM_LIFT1 = 2
    DOF_ARM_LIFT2 = 3
    DOF_GRIPPER_ROT = 4
    DOF_FINGER1 = 5
    DOF_FINGER2 = 6

    def __init__(self, gym, env, actor_handle):
        self.gym = gym
        self.env = env
        self.actor_handle = actor_handle
        self.dof_targets = np.zeros(Claw.NUM_DOFS, dtype=np.float32)
        self.cmd_queue = queue.Queue()
        self.current_cmd = None

        # get some model measurements
        base = gym.find_actor_rigid_body_handle(env, actor_handle, "base")
        finger = gym.find_actor_rigid_body_handle(env, actor_handle, "finger1")
        dof_props = gym.get_actor_dof_properties(env, actor_handle)
        base_tx = gym.get_rigid_transform(env, base)
        finger_tx = gym.get_rigid_transform(env, finger)
        self.min_ext = base_tx.p.y - finger_tx.p.y
        self.max_ext = self.min_ext
        self.max_ext += dof_props["upper"][Claw.DOF_ARM_LIFT1] - dof_props["lower"][Claw.DOF_ARM_LIFT1]
        self.max_ext += dof_props["upper"][Claw.DOF_ARM_LIFT2] - dof_props["lower"][Claw.DOF_ARM_LIFT2]
        self.base_y = base_tx.p.y
        self.max_y = self.base_y - self.min_ext
        self.min_y = self.base_y - self.max_ext

        # used when sleeping
        self.wake_time = 0.0

    def _fetch_next_command(self):
        if not self.cmd_queue.empty():
            self.current_cmd = self.cmd_queue.get()
            cmd = self.current_cmd
            if isinstance(cmd, CmdSleep):
                self.wake_time = time.time() + cmd.duration
            elif isinstance(cmd, CmdArmPos):
                self.dof_targets[Claw.DOF_ARM_X] = cmd.x_target
                self.dof_targets[Claw.DOF_ARM_Z] = cmd.z_target
            elif isinstance(cmd, CmdGripperRot):
                self.dof_targets[Claw.DOF_GRIPPER_ROT] = cmd.angle_target
            elif isinstance(cmd, CmdGripperHeight):
                d = self.max_y - cmd.y_target
                self.dof_targets[Claw.DOF_ARM_LIFT1] = -0.5 * d
                self.dof_targets[Claw.DOF_ARM_LIFT2] = -0.5 * d
            elif isinstance(cmd, CmdGripperClose):
                self.dof_targets[Claw.DOF_FINGER1] = 0.16
                self.dof_targets[Claw.DOF_FINGER2] = -0.16
            elif isinstance(cmd, CmdGripperOpen):
                self.dof_targets[Claw.DOF_FINGER1] = 0.0
                self.dof_targets[Claw.DOF_FINGER2] = 0.0
            self.gym.set_actor_dof_position_targets(self.env, self.actor_handle, self.dof_targets)
        else:
            self.current_cmd = None

    def update(self):
        cmd = self.current_cmd
        if cmd is not None:
            dof_positions = self.gym.get_actor_dof_states(self.env, self.actor_handle, gymapi.STATE_POS)['pos']
            if isinstance(cmd, CmdArmPos):
                if math.isclose(self.dof_targets[Claw.DOF_ARM_X], dof_positions[Claw.DOF_ARM_X], abs_tol=0.01) and math.isclose(self.dof_targets[Claw.DOF_ARM_Z], dof_positions[Claw.DOF_ARM_Z], abs_tol=0.01):
                    self._fetch_next_command()
            elif isinstance(cmd, CmdGripperRot):
                if math.isclose(self.dof_targets[Claw.DOF_GRIPPER_ROT], dof_positions[Claw.DOF_GRIPPER_ROT], abs_tol=0.01):
                    self._fetch_next_command()
            if isinstance(cmd, CmdGripperHeight):
                if math.isclose(self.dof_targets[Claw.DOF_ARM_LIFT1], dof_positions[Claw.DOF_ARM_LIFT1], abs_tol=0.01) and math.isclose(self.dof_targets[Claw.DOF_ARM_LIFT2], dof_positions[Claw.DOF_ARM_LIFT2], abs_tol=0.01):
                    self._fetch_next_command()
            if isinstance(cmd, CmdGripperClose):
                dof_states = gym.get_actor_dof_states(self.env, self.actor_handle, gymapi.STATE_VEL)
                v1 = dof_states['vel'][Claw.DOF_FINGER1]
                v2 = dof_states['vel'][Claw.DOF_FINGER2]
                if abs(v1) < 0.01 and abs(v2) < 0.01:
                    self._fetch_next_command()
            if isinstance(cmd, CmdGripperOpen):
                if math.isclose(self.dof_targets[Claw.DOF_FINGER1], dof_positions[Claw.DOF_FINGER1], abs_tol=0.01) and math.isclose(self.dof_targets[Claw.DOF_FINGER2], dof_positions[Claw.DOF_FINGER2], abs_tol=0.01):
                    self._fetch_next_command()
            if isinstance(cmd, CmdSleep):
                if time.time() >= self.wake_time:
                    self._fetch_next_command()
        else:
            self._fetch_next_command()


color_green = gymapi.Vec3(0.2, 0.8, 0.15)  # contact force
color_red = gymapi.Vec3(0.7, 0.2, 0.15)  # contact normal

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Claw grasping example")

if args.physics_engine != gymapi.SIM_FLEX:
    print("*** Claw example only supports FleX")
    quit()

# create sim and set simulation parameters
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 100.0

# solver parameters
sim_params.flex.solver_type = 5  # Newton - PCR (GPU)
sim_params.flex.warm_start = 0.8

# collision parameters
sim_params.flex.shape_collision_margin = 0.01  # collision margin to improve stability

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load asset
asset_root = "../../assets"
asset_file = "mjcf/claw.xml"
asset_options = gymapi.AssetOptions()
asset_options.thickness = 0.001
asset_options.fix_base_link = True  # Fix base of claw in place
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# set up the env grid with 100 environments in a 10x10 grid
num_envs = 100
num_per_row = int(math.sqrt(num_envs))
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

envs = []
claws = []

# set seed for random pose initialization
np.random.seed(42)

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add model
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0., 1.5, 0.)
    pose.r = gymapi.Quat(0., 0., 0., 1.)

    ahandle = gym.create_actor(env, asset, pose, "claw", i, 1)
    dof_props = gym.get_actor_dof_properties(env, ahandle)

    # override default stiffness, damping and max motor force values for PD control
    dof_props['stiffness'].fill(5000.0)
    dof_props['damping'].fill(250.0)
    dof_props['effort'].fill(400.0)
    gym.set_actor_dof_properties(env, ahandle, dof_props)

    box_w = random.uniform(0.1, 0.2)
    box_h = 0.2
    box_d = random.uniform(0.1, 0.4)
    box_asset_options = gymapi.AssetOptions()
    box_asset_options.density = 200.
    box_asset = gym.create_box(sim, box_w, box_h, box_d, box_asset_options)

    box_pose = gymapi.Transform()
    box_pose.p = gymapi.Vec3(0., 0.3, 0.)
    box_pose.r = gymapi.Quat(0., 0., 0., 1.)
    bhandle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)

    claw = Claw(gym, env, ahandle)
    for j in range(100):
        claw.cmd_queue.put(CmdGripperHeight(claw.min_y))
        claw.cmd_queue.put(CmdGripperClose(1.5))
        claw.cmd_queue.put(CmdGripperHeight(claw.max_y))
        claw.cmd_queue.put(CmdGripperOpen())
        claw.cmd_queue.put(CmdSleep(1.0))
    claws.append(claw)

# position the viewer camera
cam_pos = gymapi.Vec3(spacing*7, 1, -2)
cam_target = gymapi.Vec3(spacing*6, 0.5, 3)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# draw contact normals for the following environments
contact_envs = envs[2:4]
draw_scale = 0.35
# draw contact forces for the following environments
force_envs = envs[3:5]

# if enabled contacts are "manually" visualized
# illustrates how a user can modify the contact information before rendering
draw_contacts_manually = False

while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    # update all claws
    for i in range(num_envs):
        claws[i].update()

    gym.clear_lines(viewer)

    # get and draw force == sum of all contact forces acting on all bodies in a chosen envs
    for env in force_envs:
        contact_forces = gym.get_env_rigid_contact_forces(env)

        env_origin = gym.get_env_origin(env)
        poses = gym.get_env_rigid_body_states(env, gymapi.STATE_POS)['pose']

        for i in range(len(contact_forces)):
            cf = contact_forces[i]
            body_pose = gymapi.Transform.from_buffer(poses[i])
            cf_v = gymapi.Vec3(cf["x"], cf["y"], cf["z"])
            vlen = max(math.sqrt(cf["x"]**2 + cf["y"]**2 + cf["z"]**2), 1e-6)
            force_scale = 0.3 / math.sqrt(vlen)
            # draw line representing contact force
            gymutil.draw_line(body_pose.p, body_pose.p + cf_v * force_scale, color_green, gym, viewer, env)

    # get contacts and draw all contact forces in the chosen envs
    for env in contact_envs:
        if draw_contacts_manually:
            # Draw contacts "manually"
            contacts = gym.get_env_rigid_contacts(env)
            env_origin = gym.get_env_origin(env)
            poses = gym.get_env_rigid_body_states(env, gymapi.STATE_POS)['pose']
            num_contacts = len(contacts)

            for contact in contacts:
                body_index = contact["body1"]
                local_pos = contact["localPos1"]
                normal = gymapi.Vec3(contact["normal"]["x"], contact["normal"]["y"], contact["normal"]["z"])
                if contact["body1"] == -1:  # It is ground plane or soft-body
                    body_index = contact["body0"]
                    local_pos = contact["localPos0"]
                    normal = normal * -1.

                # get body pose in env space
                body_pose = gymapi.Transform.from_buffer(poses[body_index])
                contact0 = body_pose.p + body_pose.r.rotate(local_pos)

                # to reduce difference in length between the smallest and the largest vectors
                force_length = draw_scale * math.sqrt(max(contact["lambda"], 0.))
                contact_force = normal * force_length
                contact1 = contact0 + contact_force
                # draw line representing contact normal
                gymutil.draw_line(contact0, contact1, color_red, gym, viewer, env)
        else:
            gym.draw_env_rigid_contacts(viewer, env, color_red, draw_scale, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
