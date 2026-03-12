#!/usr/bin/env python
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

"""
Simplified IDM inference script that works directly with LeRobot datasets.
This is easier to use if your videos are already in LeRobot format.

Usage:
    python scripts/idm_inference_simple.py \
        --checkpoint /path/to/checkpoint \
        --dataset /path/to/lerobot_dataset \
        --output-dir /path/to/output \
        --num-gpus 1

Optional: Update dataset with predictions (useful for MP4-converted datasets)
    python scripts/idm_inference_simple.py \
        --checkpoint /path/to/checkpoint \
        --dataset /path/to/lerobot_dataset \
        --output-dir /path/to/output \
        --update-dataset
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP
from gr00t.model.idm import IDM
from gr00t.utils.video import get_all_frames_and_timestamps
from tianshou.data import Batch
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, visualization will be disabled")


def collate_fn(features_list, device):
    """Collate function to batch features together."""
    batch_dict = {}
    keys = features_list[0].keys()
    for key in keys:
        if key in ["images", "view_ids"]:
            vals = [f[key] for f in features_list]
            batch_dict[key] = torch.as_tensor(np.concatenate(vals), device=device)
        elif isinstance(features_list[0][key], (list, str)):
            # Keep lists/strings as-is
            vals = [f[key] for f in features_list]
            batch_dict[key] = vals
        else:
            vals = [f[key] for f in features_list]
            batch_dict[key] = torch.as_tensor(np.stack(vals), device=device)
    return batch_dict


def get_step_data_without_video(dataset, trajectory_id, base_index):
    """Get step data without loading video (we'll load video separately for efficiency)."""
    data = {}
    dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
    # Get the data for all modalities except video
    for modality in dataset.modality_keys:
        for key in dataset.modality_keys[modality]:
            if modality == "video":
                pass  # Skip video, we'll handle it separately
            elif modality == "state" or modality == "action":
                data[key] = dataset.get_state_or_action(trajectory_id, modality, key, base_index)
            elif modality == "language":
                data[key] = dataset.get_language(trajectory_id, key, base_index)
    return data


def get_medbot_action_names():
    """Return action names for medbot."""
    return [
        "Left Cart X",
        "Left Cart Y",
        "Left Cart Z",
        "Left Rot 0",
        "Left Rot 1",
        "Left Rot 2",
        "Left Rot 3",
        "Left Rot 4",
        "Left Rot 5",
        "Left Jaw",
        "Right Cart X",
        "Right Cart Y",
        "Right Cart Z",
        "Right Rot 0",
        "Right Rot 1",
        "Right Rot 2",
        "Right Rot 3",
        "Right Rot 4",
        "Right Rot 5",
        "Right Jaw",
    ]


def plot_action_comparison(
    predicted_actions: np.ndarray,
    ground_truth_actions: np.ndarray,
    output_file: str,
    episode_id: int,
    action_names: list = None,
):
    """
    Plot predicted vs ground truth actions.

    Args:
        predicted_actions: Predicted actions (num_steps, action_horizon, action_dim)
        ground_truth_actions: Ground truth actions (num_steps, action_dim)
        output_file: Path to save the plot
        episode_id: Episode ID for the title
        action_names: Optional list of action dimension names
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping visualization (matplotlib not available)")
        return None, None

    num_steps = len(predicted_actions)

    # For visualization, we'll use the first predicted action from each step (action_horizon=0)
    pred_first_step = predicted_actions[:, 0, :]  # (num_steps, action_dim)

    # Use the minimum dimension between predicted and ground truth
    # (predicted may have padding dimensions from the model)
    gt_action_dim = ground_truth_actions.shape[-1]
    pred_action_dim = pred_first_step.shape[-1]
    action_dim = min(gt_action_dim, pred_action_dim)

    # Trim to the common dimensions
    pred_first_step = pred_first_step[:, :action_dim]
    ground_truth_actions = ground_truth_actions[:, :action_dim]

    # Create default action names if not provided
    if action_names is None:
        action_names = [f"Action Dim {i}" for i in range(action_dim)]
    elif len(action_names) > action_dim:
        action_names = action_names[:action_dim]

    # Determine grid size for subplots
    n_cols = 4
    n_rows = (action_dim + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    fig.suptitle(f"Episode {episode_id}: Predicted vs Ground Truth Actions", fontsize=16, y=0.995)

    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    time_steps = np.arange(num_steps)

    for dim in range(action_dim):
        ax = axes[dim]

        # Plot ground truth
        ax.plot(time_steps, ground_truth_actions[:, dim], label="Ground Truth", color="blue", linewidth=2, alpha=0.7)

        # Plot prediction
        ax.plot(
            time_steps, pred_first_step[:, dim], label="Predicted", color="red", linewidth=2, alpha=0.7, linestyle="--"
        )

        # Calculate MSE for this dimension
        mse = np.mean((pred_first_step[:, dim] - ground_truth_actions[:, dim]) ** 2)

        ax.set_title(f"{action_names[dim]}\nMSE: {mse:.6f}", fontsize=10)
        ax.set_xlabel("Time Step", fontsize=9)
        ax.set_ylabel("Action Value", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(action_dim, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Calculate overall metrics
    overall_mse = np.mean((pred_first_step - ground_truth_actions) ** 2)
    overall_mae = np.mean(np.abs(pred_first_step - ground_truth_actions))

    mse_xyz1 = np.mean((ground_truth_actions[:, :3] - pred_first_step[:, :3]) ** 2)
    mse_xyz2 = np.mean((ground_truth_actions[:, 10:13] - pred_first_step[:, 10:13]) ** 2)
    mse_jaw1 = np.mean((ground_truth_actions[:, 9:10] - pred_first_step[:, 9:10]) ** 2)
    mse_jaw2 = np.mean((ground_truth_actions[:, 19:20] - pred_first_step[:, 19:20]) ** 2)
    mse_rot1 = np.mean((ground_truth_actions[:, 3:9] - pred_first_step[:, 3:9]) ** 2)
    mse_rot2 = np.mean((ground_truth_actions[:, 13:19] - pred_first_step[:, 13:19]) ** 2)
    additional_info = {
        "mse_xyz_arm1": mse_xyz1,
        "mse_xyz_arm2": mse_xyz2,
        "mse_rot_arm1": mse_rot1,
        "mse_rot_arm2": mse_rot2,
        "mse_jaw_arm1": mse_jaw1,
        "mse_jaw_arm2": mse_jaw2,
    }
    return overall_mse, overall_mae, additional_info


def update_episode_in_dataset(
    dataset_path: str,
    episode_id: int,
    predicted_actions: np.ndarray,
):
    """
    Update a single episode in the LeRobot dataset parquet files with predicted actions and states.
    State is set to the same as action (representing the achieved state after executing the action).

    Args:
        dataset_path: Path to LeRobot dataset
        episode_id: Episode ID to update
        predicted_actions: Predicted actions for this episode, shape (num_steps, action_horizon, action_dim)
    """
    dataset_path = Path(dataset_path)
    data_dir = dataset_path / "data"

    # Find all parquet files
    parquet_files = sorted(data_dir.glob("**/*.parquet"))

    if not parquet_files:
        return

    # Use first action from each prediction (action_horizon=0)
    actions_to_use = predicted_actions[:, 0, :]  # Shape: (num_steps, action_dim)

    for parquet_file in parquet_files:
        # Load parquet file
        df = pd.read_parquet(parquet_file)

        # Find rows for this episode
        episode_mask = df["episode_index"] == episode_id
        episode_rows_indices = df.index[episode_mask]

        if len(episode_rows_indices) == 0:
            continue

        # Ensure we have the right number of actions
        if len(actions_to_use) != len(episode_rows_indices):
            print(
                f"      ⚠ Action count mismatch ({len(actions_to_use)} vs {len(episode_rows_indices)}), skipping update"
            )
            return

        # Update both action and state columns row by row
        # State should match action (representing achieved state after action)
        for i, row_idx in enumerate(episode_rows_indices):
            df.at[row_idx, "action"] = actions_to_use[i]
            df.at[row_idx, "observation.state"] = actions_to_use[i]

        # Create backup on first write
        backup_path = parquet_file.with_suffix(".parquet.backup")
        if not backup_path.exists():
            import shutil

            shutil.copy(parquet_file, backup_path)

        # Save updated file
        df.to_parquet(parquet_file, engine="pyarrow", index=False)
        print(f"      ✓ Updated parquet with episode {episode_id} actions & states ({len(actions_to_use)} steps)")
        return  # Episode found and updated

    print(f"      ⚠ Episode {episode_id} not found in any parquet file")


def run_inference_on_dataset(
    checkpoint_path: str,
    dataset_path: str,
    output_dir: str,
    data_config: str = "medbot",
    embodiment_tag: str = "new_embodiment",
    video_backend: str = "torchvision_av",
    max_episodes: int = None,
    batch_size: int = 16,
    device: str = "cuda:0",
    visualize: bool = True,
    observation_indices: list = None,
    update_dataset: bool = False,
):
    """
    Run inference on a LeRobot dataset and save predicted actions.

    Args:
        checkpoint_path: Path to model checkpoint
        dataset_path: Path to LeRobot dataset
        output_dir: Directory to save outputs
        data_config: Name of data configuration
        embodiment_tag: Embodiment tag for the dataset
        video_backend: Video backend to use
        max_episodes: Maximum number of episodes to process (None = all)
        batch_size: Batch size for inference
        device: Device to run inference on
        visualize: Whether to create visualization plots
        observation_indices: Optional list of frame indices to use (e.g., [0, 8]). If None, uses config default.
        update_dataset: If True, update the dataset parquet files with predicted actions
            (useful for MP4-converted datasets)
    """
    print("=" * 80)
    print("IDM Inference on LeRobot Dataset")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Data config: {data_config}")
    print(f"Video backend: {video_backend}")
    if update_dataset:
        print("Update dataset: YES (will write predictions to parquet files)")
        print("                ⚠️  Original files will be backed up with .backup extension")
    else:
        print("Update dataset: NO (predictions saved to JSON only)")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("\n[1/4] Loading model...")
    model = IDM.from_pretrained(checkpoint_path)
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    print(f"✓ Model loaded on {device}")

    # Load dataset
    print("\n[2/4] Loading dataset...")
    data_config_cls = DATA_CONFIG_MAP[data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    # Override observation_indices if specified
    if observation_indices is not None:
        print(f"  Overriding observation_indices: {observation_indices}")
        modality_configs["video"].delta_indices = observation_indices
        if "state" in modality_configs:
            modality_configs["state"].delta_indices = observation_indices
    else:
        print(f"  Using default observation_indices: {modality_configs['video'].delta_indices}")

    # Filter out color jitter for inference
    from gr00t.data.transform import ComposedModalityTransform, VideoColorJitter

    if isinstance(transforms, ComposedModalityTransform):
        filtered = [t for t in transforms.transforms if not isinstance(t, VideoColorJitter)]
        transforms.transforms = filtered

    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=EmbodimentTag(embodiment_tag),
        video_backend=video_backend,
    )

    print(f"✓ Dataset loaded: {len(dataset.trajectory_ids)} episodes")

    # Determine episodes to process
    episode_ids = dataset.trajectory_ids
    if max_episodes is not None:
        episode_ids = episode_ids[:max_episodes]

    print(f"\n[3/4] Running inference on {len(episode_ids)} episodes...")

    # Load modality config once (needed for action reconstruction)
    modality_json_path = Path(dataset_path) / "meta" / "modality.json"
    with open(modality_json_path, "r") as f:
        modality_config = json.load(f)
    action_parts = modality_config.get("action", {})
    total_action_dim = max(indices.get("end", 0) for indices in action_parts.values())

    # Process each episode
    all_results = {}
    all_mse = []
    all_mae = []
    all_additional_info = []

    for ep_idx, episode_id in enumerate(tqdm(episode_ids, desc="Episodes")):
        try:
            # Get episode data
            episode_data = dataset.get_trajectory_data(episode_id)
            num_steps = len(episode_data)

            # Load all videos for this episode once (for efficiency)
            video_data = {}
            for key in dataset.modality_keys["video"]:
                video_path = dataset.get_video_path(episode_id, key.replace("video.", ""))
                video_backend = dataset.video_backend
                video_backend_kwargs = dataset.video_backend_kwargs
                frames, whole_indices = get_all_frames_and_timestamps(
                    video_path.as_posix(), video_backend, video_backend_kwargs
                )
                video_data[key] = (frames, whole_indices)

            # Prepare all features for this episode
            all_features = []
            timestamp = episode_data["timestamp"].to_numpy()

            for step_idx in range(num_steps):
                # Get step data without video
                step_data = get_step_data_without_video(dataset, episode_id, step_idx)

                # Add video frames for this step
                for video_key in video_data:
                    frames, whole_indices = video_data[video_key]
                    # Get the indices for this step
                    step_indices = dataset.delta_indices[video_key] + step_idx
                    step_indices = np.maximum(step_indices, 0)
                    step_indices = np.minimum(step_indices, num_steps - 1)
                    # Find matching frames
                    indices = np.array(
                        [np.where(np.isclose(whole_indices, val))[0][0] for val in timestamp[step_indices]]
                    )
                    step_data[video_key] = frames[indices]

                # Apply transforms
                transformed = dataset.transforms(step_data)
                all_features.append(transformed)

            # Run inference in batches
            episode_predictions = []

            for start_idx in range(0, num_steps, batch_size):
                end_idx = min(start_idx + batch_size, num_steps)
                batch_features = all_features[start_idx:end_idx]

                # Collate features into batch
                batch_dict = collate_fn(batch_features, device)

                # Run inference
                with torch.no_grad():
                    outputs = model.get_action(batch_dict)

                # Extract predicted actions and denormalize them
                pred_actions_tensor = outputs["action_pred"].cpu()

                # Apply inverse transform to denormalize actions back to original scale
                pred_actions_denorm = dataset.transforms.unapply(Batch(action=pred_actions_tensor))

                # Reconstruct full action array from denormalized parts
                batch_size, action_horizon, _ = pred_actions_tensor.shape
                final_actions = np.zeros((batch_size, action_horizon, total_action_dim))

                for part, indices in action_parts.items():
                    action_key = f"action.{part}"
                    start_idx = indices.get("start", 0)
                    end_idx = indices.get("end", 0)

                    if action_key in pred_actions_denorm:
                        # unapply returns numpy arrays, not tensors
                        action_data = pred_actions_denorm[action_key]
                        if isinstance(action_data, torch.Tensor):
                            action_data = action_data.numpy()
                        final_actions[:, :, start_idx:end_idx] = action_data

                # Add to episode predictions
                for i in range(batch_size):
                    episode_predictions.append(final_actions[i].tolist())

            # Get ground truth actions for comparison
            # The "action" column contains arrays, so we need to stack them properly
            ground_truth_actions = episode_data["action"].to_numpy()
            if ground_truth_actions.dtype == object:
                # If dtype is object, it means we have an array of arrays - stack them
                ground_truth_actions = np.stack(ground_truth_actions)
            if len(ground_truth_actions) > len(episode_predictions):
                ground_truth_actions = ground_truth_actions[: len(episode_predictions)]

            # Save episode results
            episode_result = {
                "episode_id": int(episode_id),
                "num_steps": num_steps,
                "predicted_actions": episode_predictions,
            }

            all_results[f"episode_{episode_id:06d}"] = episode_result

            # Update dataset with predictions immediately after inference (if requested)
            if update_dataset:
                episode_predictions_array = np.array(episode_predictions)
                update_episode_in_dataset(
                    dataset_path=dataset_path,
                    episode_id=episode_id,
                    predicted_actions=episode_predictions_array,
                )

            # Save individual episode file
            output_file = Path(output_dir) / f"episode_{episode_id:06d}_predictions.json"
            with open(output_file, "w") as f:
                json.dump(episode_result, f, indent=2)

            # Generate visualization if requested
            if visualize and MATPLOTLIB_AVAILABLE:
                plot_file = Path(output_dir) / f"episode_{episode_id:06d}_comparison.png"
                pred_actions_array = np.array(episode_predictions)

                # Get action names based on data config
                if data_config == "medbot":
                    action_names = get_medbot_action_names()
                else:
                    action_names = None

                mse, mae, additional_info = plot_action_comparison(
                    predicted_actions=pred_actions_array,
                    ground_truth_actions=ground_truth_actions,
                    output_file=str(plot_file),
                    episode_id=episode_id,
                    action_names=action_names,
                )

                if mse is not None:
                    episode_result["mse"] = float(mse)
                    episode_result["mae"] = float(mae)
                    episode_result["additional_info"] = additional_info
                    all_mse.append(mse)
                    all_mae.append(mae)
                    all_additional_info.append(additional_info)
                    print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}, Additional: {additional_info}")

        except Exception as e:
            print(f"\nError processing episode {episode_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save summary
    print("\n[4/4] Saving results...")
    summary_file = Path(output_dir) / "inference_summary.json"
    # Calculate average metrics
    avg_mse = float(np.mean(all_mse)) if all_mse else None
    avg_mae = float(np.mean(all_mae)) if all_mae else None
    avg_additional_info = {}
    if all_additional_info:
        for key in all_additional_info[0].keys():
            avg_additional_info[key] = float(np.mean([m[key] for m in all_additional_info]))

    summary = {
        "checkpoint": checkpoint_path,
        "dataset": dataset_path,
        "total_episodes": len(episode_ids),
        "processed_episodes": len(all_results),
        "data_config": data_config,
        "embodiment_tag": embodiment_tag,
        "metrics": {
            "average_mse": avg_mse,
            "average_mae": avg_mae,
            "average_additional_info": avg_additional_info,
            "per_episode_mse": [float(m) for m in all_mse],
            "per_episode_mae": [float(m) for m in all_mae],
        },
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Processed {len(all_results)} episodes")
    print(f"✓ Results saved to: {output_dir}")
    print(f"✓ Summary saved to: {summary_file}")

    if avg_mse is not None:
        print("\n📊 Overall Metrics:")
        print(f"  Average MSE: {avg_mse:.6f}")
        print(f"  Average MAE: {avg_mae:.6f}")
        print(f"  MSE range: [{min(all_mse):.6f}, {max(all_mse):.6f}]")
        print(f"  MAE range: [{min(all_mae):.6f}, {max(all_mae):.6f}]")

    if update_dataset:
        print("\n✅ Dataset updated successfully!")
        print("   Original parquet files backed up with .backup extension")

    print("\n" + "=" * 80)
    print("Inference complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run IDM inference on LeRobot dataset")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory (e.g., /tmp/gr00t/checkpoint-1000)",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to LeRobot dataset directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save inference results")
    parser.add_argument("--data-config", type=str, default="medbot", help="Data configuration name (default: medbot)")
    parser.add_argument(
        "--embodiment-tag", type=str, default="new_embodiment", help="Embodiment tag (default: new_embodiment)"
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="torchvision_av",
        choices=["decord", "opencv", "torchvision_av"],
        help="Video backend (default: torchvision_av)",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None, help="Maximum number of episodes to process (default: all)"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference (default: 16)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on (default: cuda:0)")
    parser.add_argument(
        "--no-visualize", action="store_true", help="Disable visualization of predictions vs ground truth"
    )
    parser.add_argument(
        "--observation-indices",
        type=int,
        nargs="+",
        default=None,
        help="Override observation frame indices (e.g., --observation-indices 0 8). "
        "If not specified, uses config default.",
    )
    parser.add_argument(
        "--update-dataset",
        action="store_true",
        help="Update the dataset parquet files with predicted actions "
        "(useful for MP4-converted datasets with zero actions)",
    )

    args = parser.parse_args()

    run_inference_on_dataset(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        data_config=args.data_config,
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
        max_episodes=args.max_episodes,
        batch_size=args.batch_size,
        device=args.device,
        visualize=not args.no_visualize,
        observation_indices=args.observation_indices,
        update_dataset=args.update_dataset,
    )


if __name__ == "__main__":
    main()
