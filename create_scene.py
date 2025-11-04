# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation, AssetBase, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SpawnerCfg
from isaaclab.utils import configclass
from rope import RopeFactory
from isaacsim.core.utils.stage import get_current_stage
from pxr import Usd
from collections.abc import Callable
import re
import carb

##
# Pre-defined configs
##
from isaaclab_assets import UR10e_ROBOTIQ_GRIPPER_CFG  # isort:skip

import carb
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf, Usd, UsdGeom, Gf

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import UsdFileCfg
from rope import RopeFactory
import re

UR10e_ROBOTIQ_CFG = UR10e_ROBOTIQ_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
UR10e_ROBOTIQ_CFG.init_state.joint_pos["shoulder_pan_joint"] = 0.0
UR10e_ROBOTIQ_CFG.init_state.joint_pos["shoulder_lift_joint"] = -35.0/180.0*torch.pi
UR10e_ROBOTIQ_CFG.init_state.joint_pos["elbow_joint"] = 50.0/180.0*torch.pi
UR10e_ROBOTIQ_CFG.init_state.joint_pos["wrist_1_joint"] = 50.0/180.0*torch.pi
UR10e_ROBOTIQ_CFG.init_state.joint_pos["wrist_2_joint"] = 0.0


def spawn_multi_asset(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    # get stage handle
    stage = get_current_stage()

    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    # resolve prim paths for spawning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]

    for index, prim_path in enumerate(prim_paths):
        # spawn single instance
        prim :Usd.Prim = RopeFactory(1.0, translation).create(prim_path, stage)

    # set carb setting to indicate Isaac Lab's environments that different prims have been spawned
    # at varying prim paths. In this case, PhysX parser shouldn't optimize the stage parsing.
    # the flag is mainly used to inform the user that they should disable `InteractiveScene.replicate_physics`
    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/isaaclab/spawn/multi_assets", True)

    # return the prim
    return prim_utils.get_prim_at_path(prim_paths[0])


@configclass
class RopeSpawnerCfg(SpawnerCfg):
    func = spawn_multi_asset

@configclass
class RopeKnotSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # rigid object
    object: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Rope",
        spawn=RopeSpawnerCfg(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.5, 0.0)),
    )

    # articulation
    robot: ArticulationCfg = UR10e_ROBOTIQ_CFG.copy()


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot:Articulation = scene["robot"]
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            #joint_pos += torch.rand_like(joint_pos) * 0.5
            robot.set_joint_position_target(joint_pos)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        # -- generate random joint efforts
        #efforts = torch.randn_like(robot.data.joint_pos) * 0.05
        # -- apply action to the robot
        #robot.set_joint_effort_target(efforts)
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    stage = sim.stage
    # Design scene
    scene_cfg = RopeKnotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0, replicate_physics=False)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
