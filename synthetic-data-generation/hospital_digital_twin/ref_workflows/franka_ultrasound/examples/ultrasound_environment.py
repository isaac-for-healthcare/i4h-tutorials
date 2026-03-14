# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase

# NOTE(alexmillane, 2025.09.04): There is an issue with type annotation in this file.
# We cannot annotate types which require the simulation app to be started in order to
# import, because this file is used to retrieve CLI arguments, so it must be imported
# before the simulation app is started.
# TODO(alexmillane, 2025.09.04): Fix this.


class UltrasoundEnvironment(ExampleEnvironmentBase):
    name: str = "ultrasound"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.ultrasound_approach_phantom_task import UltrasoundApproachPhantomTask
        from isaaclab_arena.utils.pose import Pose

        embodiment = self.asset_registry.get_asset_by_name("franka_ultrasound")(enable_cameras=args_cli.enable_cameras)
        background = self.asset_registry.get_asset_by_name("nurec_orca")()
        phantom = self.asset_registry.get_asset_by_name("abd_phantom")()

        embodiment.set_initial_pose(
            Pose(
                position_xyz=(0.04357140184240771, -0.5031313967179238, 0.8265578542652707),
                rotation_wxyz=(0.70711, 0.0, 0.0, 0.70711),
            )
        )
        phantom.set_initial_pose(Pose(position_xyz=(0, 0, 0.95), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        table = self.asset_registry.get_asset_by_name("table_with_cover")()
        table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)(
                pos_sensitivity=0.1,
                rot_sensitivity=0.1,
            )
        else:
            teleop_device = None
        scene = Scene(assets=[background, phantom, table])

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=UltrasoundApproachPhantomTask(phantom=phantom),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--embodiment", type=str, default="franka_ultrasound")
