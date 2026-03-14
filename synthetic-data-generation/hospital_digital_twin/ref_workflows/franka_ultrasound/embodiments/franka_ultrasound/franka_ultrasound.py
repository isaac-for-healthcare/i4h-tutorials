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

# Ported from robotic_us_ext ModFrankaUltrasoundTeleopEnv

from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab_arena.assets.asset_paths import FRANKA_ULTRASOUND
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import ee_frame_pos, ee_frame_quat

##
# Robot asset
##


FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FRANKA_ULTRASOUND,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        semantic_tags=[("class", "robot")],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=400.0,
            damping=80.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

##
# Embodiment
##


@register_asset
class FrankaUltrasoundEmbodiment(EmbodimentBase):
    """Franka Panda with Realsense D405 camera and ultrasound probe.
    - Ultrasound probe - no gripper
    - High-PD actuators, gravity disabled.
    """

    name = "franka_ultrasound"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        camera_offset: Pose | None = None,
    ):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config = FrankaUltrasoundSceneCfg()
        self.camera_config = FrankaUltrasoundCameraCfg()
        self.camera_config._camera_offset = camera_offset or _DEFAULT_FRANKA_ULTRASOUND_CAMERA_OFFSET
        self.action_config = FrankaUltrasoundActionsCfg()
        self.observation_config = FrankaUltrasoundObservationsCfg()
        self.event_config = FrankaUltrasoundEventCfg()
        self.mimic_env = FrankaUltrasoundMimicEnv


_DEFAULT_FRANKA_ULTRASOUND_CAMERA_OFFSET = Pose(
    position_xyz=(1.2, -1.0, 1.8),
    rotation_wxyz=(0.328865, -0.242957, 0.087962, 0.908341),
)


@configclass
class FrankaUltrasoundSceneCfg:
    """Additions to the scene configuration coming from the FrankaUltrasound embodiment."""

    # Franka with ultrasound probe — high-PD gains, gravity disabled, no gripper fingers
    robot: ArticulationCfg = FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # End-effector frame — tracks the TCP (probe tip) relative to panda_link0
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=True,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/TCP",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
        ],
    )

    def __post_init__(self):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


@configclass
class FrankaUltrasoundCameraCfg:
    """Camera configuration for the ultrasound scene."""

    room_cam: CameraCfg = MISSING

    def __post_init__(self):
        camera_offset = getattr(self, "_camera_offset", _DEFAULT_FRANKA_ULTRASOUND_CAMERA_OFFSET)

        self.room_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/room_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0,
                focus_distance=100.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=camera_offset.position_xyz,
                rot=camera_offset.rotation_wxyz,
                convention="world",
            ),
        )


@configclass
class FrankaUltrasoundActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="TCP",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )


@configclass
class FrankaUltrasoundObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state and camera values."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


##
# Event functions
##


def _reset_panda_joints_by_fraction_of_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fraction: float = 0.1,
):
    """Reset the robot joints with offsets sampled from a fraction of the joint limits."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    joint_limits = asset.data.default_joint_limits[env_ids].clone()
    joint_sample_ranges = joint_limits * fraction

    lower = joint_sample_ranges[:, :, 0]
    upper = joint_sample_ranges[:, :, 1]

    joint_pos_delta = torch.rand(joint_pos.shape, device=joint_pos.device) * (upper - lower) + lower
    joint_pos = torch.clamp(joint_pos + joint_pos_delta, joint_limits[:, :, 0], joint_limits[:, :, 1])

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


@configclass
class FrankaUltrasoundEventCfg:
    """Configuration for FrankaUltrasound."""

    reset_joint_position = EventTerm(
        func=_reset_panda_joints_by_fraction_of_limits,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "fraction": 0.01,
        },
    )


##
# Mimic
##


class FrankaUltrasoundMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic env for the Franka ultrasound embodiment (no gripper, 6-DOF action space)."""

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current end-effector pose as a (N, 4, 4) matrix."""
        if env_ids is None:
            env_ids = slice(None)
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        noise: float | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert a target EEF pose to a 6-DOF delta-pose action (no gripper)."""
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        delta_position = target_pos - curr_pos

        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if noise is not None:
            pose_action = pose_action + noise * torch.randn_like(pose_action)
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return pose_action

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert a 6-DOF delta-pose action back to a target EEF pose."""
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        target_pos = curr_pos + delta_position

        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        is_close_to_zero = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_close_to_zero] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        return {eef_name: PoseUtils.make_pose(target_pos, target_rot).clone()}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """No gripper — returns empty tensors."""
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        return {eef_name: torch.zeros(actions.shape[0], 0, device=actions.device)}

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """Get pose of every rigid/articulated object in the scene."""
        if env_ids is None:
            env_ids = slice(None)
        state = self.scene.get_state(is_relative=True)
        return get_rigid_and_articulated_object_poses(state, env_ids)

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Read subtask completion signals from obs_buf["subtask_terms"].

        The signals are defined by the task via get_observation_cfg() — the robot
        embodiment has no knowledge of what each signal represents.
        """
        if env_ids is None:
            env_ids = slice(None)
        return {key: val[env_ids] for key, val in self.obs_buf["subtask_terms"].items()}
