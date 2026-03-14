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
import math
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import configclass
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


def _probe_near_point(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    phantom_cfg: SceneEntityCfg,
    point_offset: tuple[float, float, float],
    threshold: float,
) -> torch.Tensor:
    """Returns True when the probe EEF is within ``threshold`` metres of a phantom-local point."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    phantom: RigidObject = env.scene[phantom_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 0, :]

    phantom_pos = phantom.data.root_pos_w
    phantom_quat = phantom.data.root_quat_w
    offset = torch.tensor(point_offset, device=env.device).expand(env.num_envs, -1)
    target_pos = phantom_pos + math_utils.quat_apply(phantom_quat, offset)

    dist = torch.norm(ee_pos - target_pos, dim=-1)
    return dist < threshold


def probe_reach_phantom(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    phantom_cfg: SceneEntityCfg = SceneEntityCfg("abd_phantom"),
    contact_point_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    threshold: float = 0.05,
) -> torch.Tensor:
    """Returns True when the probe EEF is within ``threshold`` metres of the phantom scan start point."""
    result = _probe_near_point(env, ee_frame_cfg, phantom_cfg, contact_point_offset, threshold)
    return result


def probe_scan_phantom(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    phantom_cfg: SceneEntityCfg = SceneEntityCfg("abd_phantom"),
    scan_end_point_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    threshold: float = 0.05,
) -> torch.Tensor:
    """Returns True when the probe EEF is within ``threshold`` metres of the phantom scan end point."""
    result = _probe_near_point(env, ee_frame_cfg, phantom_cfg, scan_end_point_offset, threshold)
    return result


class UltrasoundApproachPhantomTask(TaskBase):
    """Task for approaching and scanning an ultrasound phantom with the probe end-effector.

    Two subtasks:
      1. reach_phantom — bring the probe to the scan contact point on the phantom surface.
      2. scan_phantom  — slide the probe from the contact point to the scan end point.

    On each reset the phantom is randomly displaced in X/Y/yaw so the robot must
    re-approach and re-scan it.  The episode succeeds when the EEF is within
    ``threshold`` metres of the phantom's scan end point.
    """

    def __init__(
        self,
        phantom: Asset,
        episode_length_s: float | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.phantom = phantom

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        success = TerminationTermCfg(
            func=probe_scan_phantom,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "phantom_cfg": SceneEntityCfg(self.phantom.name),
                "scan_end_point_offset": self.phantom.scan_end_point_offset,
            },
        )
        return TerminationsCfg(success=success)

    def get_events_cfg(self):
        return EventsCfg(self.phantom)

    def get_prompt(self) -> str:
        return "Approach the ultrasound phantom with the probe and scan from start to end point."

    def get_mimic_env_cfg(self, embodiment_name: str):
        return UltrasoundApproachPhantomMimicEnvCfg(
            embodiment_name=embodiment_name,
            phantom_name=self.phantom.name,
        )

    def get_observation_cfg(self):
        subtask_terms = UltrasoundSubtaskTermsCfg(
            reach_1=ObsTerm(
                func=probe_reach_phantom,
                params={
                    "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                    "phantom_cfg": SceneEntityCfg(self.phantom.name),
                    "contact_point_offset": self.phantom.contact_point_offset,
                },
            )
        )
        return UltrasoundTaskObservationsCfg(subtask_terms=subtask_terms)

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.phantom,
            offset=np.array([0.6, 0.3, 1.0]),
        )


@configclass
class TerminationsCfg:
    """Termination terms for the ultrasound approach task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING


@configclass
class EventsCfg:
    """Events for the ultrasound approach task."""

    reset_phantom_pose: EventTermCfg = MISSING

    def __init__(self, phantom: Asset):
        self.reset_phantom_pose = EventTermCfg(
            func=mdp_isaac_lab.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.025, 0.025),
                    "y": (0.10, 0.12),
                    "z": (-0.0, -0.0),
                    "yaw": (-math.pi / 8, math.pi / 8),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg(phantom.name),
            },
        )


@configclass
class UltrasoundApproachPhantomMimicEnvCfg(MimicEnvCfg):
    """Isaac Lab Mimic environment config for the ultrasound scan task (two subtasks)."""

    embodiment_name: str = "franka_ultrasound"
    phantom_name: str = "abd_phantom"

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "demo_src_ultrasound_approach_phantom"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = False
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        subtask_configs = [
            # Subtask 1: reach the scan contact point on the phantom surface.
            # "reach_1" signal fires when the probe first touches the contact point.
            # Annotator presses S at this moment during manual annotation.
            SubTaskConfig(
                object_ref=self.phantom_name,
                subtask_term_signal="reach_1",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.001,
                num_interpolation_steps=5,
                num_fixed_steps=1,
                apply_noise_during_interpolation=False,
            ),
            # Subtask 2: slide from contact point to scan end point.
            # Final subtask — no term signal needed.
            SubTaskConfig(
                object_ref=self.phantom_name,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.001,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            ),
        ]
        self.subtask_configs["robot"] = subtask_configs


##
# Task-level observations — subtask completion signals
##


@configclass
class UltrasoundSubtaskTermsCfg(ObsGroup):
    """Subtask completion signals for the ultrasound scan task.

    reach_1: True when the probe EEF is within threshold of the phantom scan contact point.
    """

    reach_1: ObsTerm = MISSING

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class UltrasoundTaskObservationsCfg:
    """Task-level observation groups for the ultrasound scan task."""

    subtask_terms: UltrasoundSubtaskTermsCfg = MISSING
