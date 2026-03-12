#!/usr/bin/env python3
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
Script to organize video data into robotic_us_example folder structure.
Creates separate timestamped output directories for each episode.
Generates configs for both room camera and wrist camera.

Usage:
    # Output defaults to parent of source directory
    ./organize_video_data.py --source /ephemeral/videos

    # Or specify output directory explicitly
    ./organize_video_data.py --source /workspace/videos --output /workspace/robotic_us_example
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def get_video_resolution(video_path):
    """
    Get the resolution of a video file.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height) or None if video cannot be read
    """
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return (width, height)


def resize_label_masks_to_720p(label_frames):
    """
    Resize label mask frames to 720p (1280x720) using nearest neighbor interpolation.
    Handles aspect ratio by padding with background label (0) if needed.

    Args:
        label_frames: (T, H, W) array with integer labels

    Returns:
        (T, 720, 1280) array with resized integer labels
    """
    print("    Resizing masks to 720p (1280x720) using nearest neighbor...")
    print(f"      Input shape: {label_frames.shape}")

    num_frames = label_frames.shape[0]
    src_height, src_width = label_frames.shape[1], label_frames.shape[2]

    target_width, target_height = 1280, 720

    # Calculate scaling to fit within 1280x720 while preserving aspect ratio
    scale_w = target_width / src_width
    scale_h = target_height / src_height
    scale = min(scale_w, scale_h)

    new_width = int(src_width * scale)
    new_height = int(src_height * scale)

    print(f"      Scaled size: {new_width}x{new_height}")

    # Calculate padding
    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left
    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top

    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        print(f"      Padding: left={pad_left}, right={pad_right}, top={pad_top}, bottom={pad_bottom}")

    resized_frames = np.zeros((num_frames, target_height, target_width), dtype=np.uint8)

    for i in range(num_frames):
        # Resize using nearest neighbor to preserve label values
        resized = cv2.resize(label_frames[i], (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # Pad with background label (0)
        resized_frames[i, pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized

    print(f"      Output shape: {resized_frames.shape}")
    print(f"      Label range: [{resized_frames.min()}, {resized_frames.max()}]")
    print(f"      Unique labels: {np.unique(resized_frames)}")

    return resized_frames


def convert_rgb_mask_to_labels(rgb_frames):
    """
    Convert RGB mask frames to integer labels using hardcoded color mapping.

    Args:
        rgb_frames: (T, H, W, 3) array with RGB values

    Returns:
        (T, H, W) array with integer labels 0-4
    """
    # Hardcoded color mapping: RGB -> Label
    color_to_label = {
        (0, 0, 0): 0,  # Black (Background) -> 0
        (255, 0, 0): 1,  # Red -> 1
        (0, 0, 255): 2,  # Blue -> 2
        (0, 255, 0): 3,  # Green -> 3
        (255, 255, 0): 4,  # Yellow -> 4
    }

    # Create output array
    label_frames = np.zeros(rgb_frames.shape[:-1], dtype=np.uint8)

    # Map each pixel
    for rgb, label in color_to_label.items():
        mask = np.all(rgb_frames == rgb, axis=-1)
        label_frames[mask] = label

    return label_frames


def create_folder_structure(base_dir):
    """Create the folder structure matching robotic_us_example."""
    folders = [
        "depth",
        "edge",
        "multicontrol",
        "outputs",
        "robotic_us",
        "seg",
    ]

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created: {folder_path}")


def copy_and_rename_video_data(source_dir, dest_base_dir, episode_filter=None):
    """
    Copy video data from source to destination, renaming and organizing files.

    Mapping:
    - rgb_video_0.mp4 → robotic_us_input.mp4 (root)
    - rgb_video_1.mp4 → robotic_us_wrist_input.mp4 (root)
    - depth_video_0.mp4 → depth/robotic_us_depth.mp4
    - depth_video_1.mp4 → depth/robotic_us_wrist_depth.mp4
    - seg_mask_video_0.mp4 → seg/robotic_us_seg.mp4
    - seg_mask_video_1.mp4 → seg/robotic_us_wrist_seg.mp4
    - seg_masks.npz → seg/robotic_us_mask.npz (converted from RGB to labels 0-4)
    - seg_masks.npz → seg/robotic_us_wrist_mask.npz (converted from RGB to labels 0-4)
    - room_camera_para.npz → seg/room_camera_para.npz
    - wrist_camera_para.npz → seg/wrist_camera_para.npz
    - Original data_X folders also kept in robotic_us/data_X/

    Args:
        source_dir: Directory containing data_X subdirectories
        dest_base_dir: Base output directory
        episode_filter: If specified, only process this episode folder (e.g., 'data_0')

    Note: RGB masks are converted using hardcoded mapping:
      Black(0,0,0)→0, Red(255,0,0)→1, Blue(0,0,255)→2, Green(0,255,0)→3, Yellow(255,255,0)→4
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        print("\nTip: Make sure to specify the correct source directory:")
        print("  ./organize_video_data.py --source /workspace/robotic_us_example/videos")
        return []

    # Track which data folder we're processing for mask naming
    data_folders = sorted([f for f in source_path.iterdir() if f.is_dir() and f.name.startswith("data_")])

    # Filter to specific episode if requested
    if episode_filter:
        data_folders = [f for f in data_folders if f.name == episode_filter]
        if not data_folders:
            print(f"Error: Episode folder '{episode_filter}' not found in {source_dir}")
            return []

    processed_episodes = []

    # Process each data_X folder
    for data_folder in data_folders:
        print(f"\nProcessing {data_folder.name}...")

        # Copy rgb videos to root with new names
        rgb_0 = data_folder / "rgb_video_0.mp4"
        rgb_1 = data_folder / "rgb_video_1.mp4"

        # Detect video resolution to determine if masks need resizing
        is_720p = False
        if rgb_0.exists():
            dest = os.path.join(dest_base_dir, "robotic_us_input.mp4")
            shutil.copy2(rgb_0, dest)
            print("  Copied: rgb_video_0.mp4 → robotic_us_input.mp4")

            # Check resolution
            resolution = get_video_resolution(rgb_0)
            if resolution:
                width, height = resolution
                print(f"    Detected resolution: {width}x{height}")
                if width == 1280 and height == 720:
                    is_720p = True
                    print("    → 720p detected, masks will be resized to match")

        if rgb_1.exists():
            dest = os.path.join(dest_base_dir, "robotic_us_wrist_input.mp4")
            shutil.copy2(rgb_1, dest)
            print("  Copied: rgb_video_1.mp4 → robotic_us_wrist_input.mp4")

        # Copy depth videos to depth/ with new names
        depth_0 = data_folder / "depth_video_0.mp4"
        depth_1 = data_folder / "depth_video_1.mp4"
        if depth_0.exists():
            dest = os.path.join(dest_base_dir, "depth", "robotic_us_depth.mp4")
            shutil.copy2(depth_0, dest)
            print("  Copied: depth_video_0.mp4 → depth/robotic_us_depth.mp4")
        if depth_1.exists():
            dest = os.path.join(dest_base_dir, "depth", "robotic_us_wrist_depth.mp4")
            shutil.copy2(depth_1, dest)
            print("  Copied: depth_video_1.mp4 → depth/robotic_us_wrist_depth.mp4")

        # Copy seg mask videos to seg/ with new names
        seg_0 = data_folder / "seg_mask_video_0.mp4"
        seg_1 = data_folder / "seg_mask_video_1.mp4"
        if seg_0.exists():
            dest = os.path.join(dest_base_dir, "seg", "robotic_us_seg.mp4")
            shutil.copy2(seg_0, dest)
            print("  Copied: seg_mask_video_0.mp4 → seg/robotic_us_seg.mp4")
        if seg_1.exists():
            dest = os.path.join(dest_base_dir, "seg", "robotic_us_wrist_seg.mp4")
            shutil.copy2(seg_1, dest)
            print("  Copied: seg_mask_video_1.mp4 → seg/robotic_us_wrist_seg.mp4")

        # Convert and copy segmentation mask file to seg/ for guided generation
        seg_masks = data_folder / "seg_masks.npz"
        if seg_masks.exists():
            print("  Converting seg_masks.npz from RGB to labels...")

            # Load the multi-camera RGB masks
            data = np.load(seg_masks)
            if "seg_images" in data:
                all_frames = data["seg_images"]  # Shape: (T, N_cameras, H, W, 3)
            elif "arr_0" in data:
                all_frames = data["arr_0"]
            else:
                print("    Warning: Unknown npz format, skipping conversion")
                continue

            # Convert both cameras (room camera = camera 0, wrist camera = camera 1)
            # Room camera (camera 0)
            if all_frames.shape[1] > 0:
                rgb_frames_cam0 = all_frames[:, 0, :, :, :]
                label_frames_cam0 = convert_rgb_mask_to_labels(rgb_frames_cam0)

                # Resize to 720p if video is 720p
                if is_720p:
                    label_frames_cam0 = resize_label_masks_to_720p(label_frames_cam0)

                dest = os.path.join(dest_base_dir, "seg", "robotic_us_mask.npz")
                np.savez(dest, arr_0=label_frames_cam0)
                print("    Converted camera 0: seg_masks.npz → seg/robotic_us_mask.npz")
                print(f"      Shape: {label_frames_cam0.shape}, Labels: {np.unique(label_frames_cam0)}")

            # Wrist camera (camera 1) if exists
            if all_frames.shape[1] > 1:
                rgb_frames_cam1 = all_frames[:, 1, :, :, :]
                # Check if camera 1 is not empty
                if not np.all(rgb_frames_cam1 == 0):
                    label_frames_cam1 = convert_rgb_mask_to_labels(rgb_frames_cam1)

                    # Resize to 720p if video is 720p
                    if is_720p:
                        label_frames_cam1 = resize_label_masks_to_720p(label_frames_cam1)

                    dest = os.path.join(dest_base_dir, "seg", "robotic_us_wrist_mask.npz")
                    np.savez(dest, arr_0=label_frames_cam1)
                    print("    Converted camera 1: seg_masks.npz → seg/robotic_us_wrist_mask.npz")
                    print(f"      Shape: {label_frames_cam1.shape}, Labels: {np.unique(label_frames_cam1)}")
                else:
                    print("    Skipped camera 1: empty (all black pixels)")

        # Also copy camera parameter npz files to seg/
        room_cam = data_folder / "room_camera_para.npz"
        wrist_cam = data_folder / "wrist_camera_para.npz"
        if room_cam.exists():
            dest = os.path.join(dest_base_dir, "seg", "room_camera_para.npz")
            shutil.copy2(room_cam, dest)
            print("  Copied: room_camera_para.npz → seg/room_camera_para.npz")
        if wrist_cam.exists():
            dest = os.path.join(dest_base_dir, "seg", "wrist_camera_para.npz")
            shutil.copy2(wrist_cam, dest)
            print("  Copied: wrist_camera_para.npz → seg/wrist_camera_para.npz")

        # Also keep original folder structure in robotic_us/data_X/
        dest_folder = os.path.join(dest_base_dir, "robotic_us", data_folder.name)
        os.makedirs(dest_folder, exist_ok=True)

        for file_path in data_folder.iterdir():
            if file_path.is_file():
                dest_file = os.path.join(dest_folder, file_path.name)
                shutil.copy2(file_path, dest_file)
            elif file_path.is_dir():
                dest_subdir = os.path.join(dest_folder, file_path.name)
                shutil.copytree(file_path, dest_subdir, dirs_exist_ok=True)

        print(f"  Copied: All files → robotic_us/{data_folder.name}/")
        processed_episodes.append(data_folder.name)

    return processed_episodes


def create_prompt_file(dest_base_dir):
    """
    Create prompt text file for video generation.
    """
    prompt_text = (
        "A robotic ultrasound procedure in a modern hospital operating room. "
        "A robotic arm moves an ultrasound probe over a medical phantom placed on a hospital bed. "
        "The bed is a standard white medical examination table with adjustable height. "
        "The room has the sterile appearance of a surgical suite with clean white walls, "
        "medical equipment cabinets, monitoring screens, and overhead surgical lights. "
        "The floor is tiled in light colors typical of hospital environments. "
        "Medical instruments and supplies are visible on rolling carts nearby. "
        "The robotic system performs precise scanning movements across the phantom surface "
        "while maintaining proper contact. The overall atmosphere is professional and clinical, "
        "with bright even lighting characteristic of a hospital operating room."
    )

    prompt_path = os.path.join(dest_base_dir, "robotic_us_prompt.txt")
    with open(prompt_path, "w") as f:
        f.write(prompt_text.strip() + "\n")

    print("Created prompt file: robotic_us_prompt.txt")
    return prompt_path


def create_wrist_prompt_file(dest_base_dir):
    """
    Create prompt text file for wrist camera video generation.
    """
    prompt_text = (
        "Close-up view of a robotic ultrasound probe scanning a medical phantom. "
        "The ultrasound probe is held by a robotic arm and moves in precise contact with the phantom surface. "
        "The scene shows the probe tip, phantom surface, and immediate workspace with clinical lighting. "
        "Sterile, professional medical environment with the probe and phantom in focus. "
        "The wrist camera is facing down towards the table and the floor at most times."
    )
    prompt_path = os.path.join(dest_base_dir, "robotic_us_wrist_prompt.txt")
    with open(prompt_path, "w") as f:
        f.write(prompt_text.strip() + "\n")
    print("Created wrist prompt file: robotic_us_wrist_prompt.txt")
    return prompt_path


def create_control_config(dest_base_dir, config_name="robotic_us_multicontrol_guided"):
    """
    Create multicontrol guided generation config JSON file for room camera.
    References the room camera files:
    - robotic_us_input.mp4 (from rgb_video_0.mp4)
    - robotic_us_depth.mp4 (from depth_video_0.mp4)
    - robotic_us_seg.mp4 (from seg_mask_video_0.mp4)
    - robotic_us_mask.npz (from seg_masks.npz in data_0)
    """
    config = {
        "name": config_name,
        "prompt_path": "../robotic_us_prompt.txt",
        "video_path": "../robotic_us_input.mp4",
        "guided_generation_mask": "../seg/robotic_us_mask.npz",
        "guided_generation_step_threshold": 25,
        "guided_generation_foreground_labels": [2, 4],
        "guidance": 3,
        "depth": {"control_path": "../depth/robotic_us_depth.mp4", "control_weight": 0.5},
        "seg": {"control_path": "../seg/robotic_us_seg.mp4", "control_weight": 0.5},
        "vis": {"control_path": None, "control_weight": 0.0},
        "edge": {"control_path": None, "control_weight": 0.0},
    }

    # Write config to multicontrol folder
    config_path = os.path.join(dest_base_dir, "multicontrol", f"{config_name}_spec.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created control config: multicontrol/{config_name}_spec.json")
    return config_path


def create_wrist_control_config(dest_base_dir, config_name="robotic_us_wrist_multicontrol_guided"):
    """
    Create multicontrol guided generation config JSON file for wrist camera.
    References the wrist camera files:
    - robotic_us_wrist_input.mp4 (from rgb_video_1.mp4)
    - robotic_us_wrist_depth.mp4 (from depth_video_1.mp4)
    - robotic_us_wrist_seg.mp4 (from seg_mask_video_1.mp4)
    - robotic_us_wrist_mask.npz (from seg_masks.npz in data_1)
    """
    config = {
        "name": config_name,
        "prompt_path": "../robotic_us_wrist_prompt.txt",
        "video_path": "../robotic_us_wrist_input.mp4",
        "guided_generation_mask": "../seg/robotic_us_wrist_mask.npz",
        "guided_generation_step_threshold": 25,
        "guided_generation_foreground_labels": [2, 4],
        "guidance": 3,
        "depth": {"control_path": "../depth/robotic_us_wrist_depth.mp4", "control_weight": 0.5},
        "seg": {"control_path": "../seg/robotic_us_wrist_seg.mp4", "control_weight": 0.5},
        "vis": {"control_path": None, "control_weight": 0.0},
        "edge": {"control_path": None, "control_weight": 0.0},
    }

    # Write config to multicontrol folder
    config_path = os.path.join(dest_base_dir, "multicontrol", f"{config_name}_spec.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Created wrist control config: multicontrol/{config_name}_spec.json")
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description="Organize video data into robotic_us_example folder structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simplest usage - output defaults to parent of source
  ./organize_video_data.py --source /workspace/robotic_us_example/videos

  # Specify output directory explicitly
  ./organize_video_data.py --source /workspace/videos --output /workspace/robotic_us_example

  # Process only a specific episode
  ./organize_video_data.py --source /workspace/videos --episode data_0

  # Full example with all options
  ./organize_video_data.py --source /workspace/videos --output /workspace/output --episode data_1

Note: If --output is not specified, it defaults to the parent directory of --source
        """,
    )

    parser.add_argument(
        "--source", "-s", type=str, required=True, help="Source directory containing video data (REQUIRED)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output base directory (default: parent directory of source)"
    )
    parser.add_argument(
        "--episode",
        "-e",
        type=str,
        default=None,
        help="Process only specific episode (e.g., 'data_0'). If not specified, processes all episodes.",
    )

    args = parser.parse_args()

    # If no output directory specified, use parent of source directory
    if not args.output:
        source_parent = str(Path(args.source).parent)
        args.output = source_parent
        print(f"No output directory specified. Using parent of source: {args.output}\n")

    # Find all episode folders
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source directory {args.source} does not exist")
        return

    data_folders = sorted([f for f in source_path.iterdir() if f.is_dir() and f.name.startswith("data_")])

    # Filter to specific episode if requested
    if args.episode:
        data_folders = [f for f in data_folders if f.name == args.episode]
        if not data_folders:
            print(f"Error: Episode '{args.episode}' not found in {args.source}")
            return

    if not data_folders:
        print(f"Error: No data_X folders found in {args.source}")
        return

    print(f"Found {len(data_folders)} episode(s) to process: {[f.name for f in data_folders]}\n")

    # Process each episode into its own organized folder
    all_output_dirs = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for data_folder in data_folders:
        episode_name = data_folder.name  # e.g., 'data_0'
        output_dir_name = f"robotic_us_organized_{timestamp}_{episode_name}"

        # Base directory for this episode's output
        base_output_dir = os.path.join(args.output, output_dir_name)

        print(f"{'='*80}")
        print(f"Processing episode: {episode_name}")
        print(f"Creating organized data structure in: {base_output_dir}\n")

        # Create base directory
        os.makedirs(base_output_dir, exist_ok=True)

        # Create folder structure
        print("Creating folder structure...")
        create_folder_structure(base_output_dir)

        # Copy and rename video data for this episode only
        print("\nCopying and renaming video data...")
        processed = copy_and_rename_video_data(args.source, base_output_dir, episode_filter=episode_name)

        if not processed:
            print(f"Warning: No data processed for {episode_name}")
            continue

        # Create prompt files and control configs
        print("\nCreating prompt and control configs...")
        create_prompt_file(base_output_dir)
        create_wrist_prompt_file(base_output_dir)
        create_control_config(base_output_dir)
        create_wrist_control_config(base_output_dir)

        all_output_dirs.append(base_output_dir)

        print(f"\n✓ Episode {episode_name} organization complete!")

    # Print summary
    print(f"\n{'='*80}")
    print("✓ All episodes processed successfully!")
    print(f"\nCreated {len(all_output_dirs)} organized directories:")
    for output_dir in all_output_dirs:
        print(f"  - {output_dir}")

    print("\nEach directory contains:")
    print("  ├── robotic_us_input.mp4 (room camera RGB)")
    print("  ├── robotic_us_wrist_input.mp4 (wrist camera RGB)")
    print("  ├── robotic_us_prompt.txt (room camera)")
    print("  ├── robotic_us_wrist_prompt.txt (wrist camera)")
    print("  ├── depth/")
    print("  │   ├── robotic_us_depth.mp4")
    print("  │   └── robotic_us_wrist_depth.mp4")
    print("  ├── multicontrol/")
    print("  │   ├── robotic_us_multicontrol_guided_spec.json (room camera)")
    print("  │   └── robotic_us_wrist_multicontrol_guided_spec.json (wrist camera)")
    print("  ├── outputs/ (for generation results)")
    print("  ├── robotic_us/")
    print("  │   └── data_X/ (original files)")
    print("  └── seg/")
    print("      ├── robotic_us_seg.mp4")
    print("      ├── robotic_us_wrist_seg.mp4")
    print("      ├── robotic_us_mask.npz (room camera masks)")
    print("      ├── robotic_us_wrist_mask.npz (wrist camera masks)")
    print("      ├── room_camera_para.npz")
    print("      └── wrist_camera_para.npz")


if __name__ == "__main__":
    main()
