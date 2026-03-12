# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import numpy as np
import tyro
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

python scripts/eval_policy.py --plot --model-path nvidia/GR00T-N1.5-3B
"""


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "localhost"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    plot: bool = False
    """Whether to plot the images."""

    modality_keys: List[str] = field(
        default_factory=lambda: [
            "left_cartesian",
            "left_rotation",
            "left_jaw",
            "right_cartesian",
            "right_rotation",
            "right_jaw",
        ]
    )
    """Modality keys to evaluate."""

    data_config: str = "fourier_gr1_arms_only"
    """
    Data config to use, e.g. so100, fourier_gr1_arms_only, unitree_g1, etc.
    Or a path to a custom data config file. e.g. "module:ClassName" format.
    See gr00t/experiment/data_config.py for more details.
    """

    steps: int = None
    """Number of steps to evaluate. If None, will use the actual trajectory length."""

    trajs: int = None
    """Number of trajectories to evaluate. If None, will evaluate all trajectories in the dataset."""

    action_horizon: int = None
    """Action horizon to evaluate. If None, will use the data config's action horizon."""

    video_backend: Literal["decord", "torchvision_av", "torchcodec"] = "torchcodec"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""

    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    """Path to the dataset."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """Embodiment tag to use."""

    model_path: str = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str = None
    """Path to save the plot."""

    plot_state: bool = False
    """Whether to plot the state."""


def main(args: ArgsConfig):
    data_config = load_data_config(args.data_config)

    # Set action_horizon from data config if not provided
    if args.action_horizon is None:
        args.action_horizon = len(data_config.action_indices)
        print(f"Using action_horizon={args.action_horizon} from data config '{args.data_config}'")

    if args.model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print("Current modality config: \n", modality)

    # Create the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
    )

    print(len(dataset))
    # Make a prediction
    obs = dataset[0]
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    for k, v in dataset.get_step_data(0, 0).items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    print("Total trajectories:", len(dataset.trajectory_lengths))
    print("All trajectories:", dataset.trajectory_lengths)
    print("Running on all trajs with modality keys:", args.modality_keys)

    # If trajs is not specified, evaluate all trajectories
    num_trajs = args.trajs if args.trajs is not None else len(dataset.trajectory_lengths)
    print(f"Evaluating {num_trajs} trajectories")

    all_mse = []
    additional_mse = []
    trajectory_results = []

    for traj_id in range(num_trajs):
        # Use actual trajectory length if steps is not specified
        traj_steps = args.steps if args.steps is not None else dataset.trajectory_lengths[traj_id]
        print(f"Running trajectory {traj_id} with {traj_steps} steps")

        mse, additional_info = calc_mse_for_single_trajectory(
            policy,
            dataset,
            traj_id,
            modality_keys=args.modality_keys,
            steps=traj_steps,
            action_horizon=args.action_horizon,
            plot=args.plot,
            plot_state=args.plot_state,
            save_plot_path=args.save_plot_path.replace(".png", f"_traj{traj_id}.png")
            if args.save_plot_path is not None
            else None,
        )
        print("MSE:", mse)
        all_mse.append(mse)
        if "gt_action_across_time" in additional_info:
            del additional_info["pred_action_across_time"]
            del additional_info["gt_action_across_time"]
        additional_mse.append(additional_info)
        # Store trajectory-level results (includes MSE for each component)
        trajectory_results.append(
            {
                "trajectory_id": int(traj_id),
                "steps": int(traj_steps),
                "mse": float(mse),
                **additional_info,  # Include all component MSE values
            }
        )

    # Calculate mean and std across all trajectories
    average_mse = float(np.mean(all_mse))
    std_mse = float(np.std(all_mse))

    addditional_average_info = {}
    for key in additional_mse[0].keys():
        mse_values = [m[key] for m in additional_mse]
        addditional_average_info[key] = float(np.mean(mse_values))
        addditional_average_info[f"std_{key.replace('mse_', '')}"] = float(np.std(mse_values))

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Average MSE: {average_mse:.6f} ± {std_mse:.6f}")
    print("\nComponent MSE (mean ± std across trajectories):")
    mse_keys = [k for k in addditional_average_info.keys() if k.startswith("mse_")]
    for mse_key in mse_keys:
        std_key = f"std_{mse_key.replace('mse_', '')}"
        if std_key in addditional_average_info:
            print(f"  {mse_key}: {addditional_average_info[mse_key]:.6f} ± {addditional_average_info[std_key]:.6f}")
        else:
            print(f"  {mse_key}: {addditional_average_info[mse_key]:.6f}")

    # Save results to JSON file
    if args.save_plot_path is not None:
        json_path = Path(args.save_plot_path).with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {
            "average_mse": average_mse,
            "std_mse": std_mse,
            "average_component_metrics": addditional_average_info,  # Includes both mean and std for each component
            "num_trajectories": int(num_trajs),
            "modality_keys": args.modality_keys,
            "action_horizon": int(args.action_horizon) if args.action_horizon is not None else None,
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "trajectories": trajectory_results,
        }

        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {json_path}")

    print("Done")
    exit()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
