# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script for extracting frames or videos from HDF5 files based on split file.

Usage:
python extract_frames_videos.py \
    --data_dir ./dataset/ \
    --split_file ./dataset/data_split.json \
    --splits test_B \
    --output_dir ./output/ \
    --mode frame  # or 'video'
    --camera_key cam_high
"""

import argparse
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path

import h5py
import tqdm
from PIL import Image


def load_episodes_from_split(split_file: str, splits: list):
    """
    Load episodes from train/test split JSON file.

    Parameters:
    - split_file: Path to the JSON file with train/test split
    - splits: List of split keys to combine (e.g., ['train1', 'train2'] or ['test1'])

    Returns:
    - List of full file paths
    """
    with open(split_file, "r") as f:
        split_data = json.load(f)

    # Validate all requested splits exist
    available_splits = list(split_data.keys())
    for split in splits:
        if split not in available_splits:
            raise ValueError(f"Split '{split}' not found. Available splits: {available_splits}")

    # Combine all requested splits
    episodes = []
    for split in splits:
        episodes.extend(split_data[split])

    return episodes


def extract_frame_from_hdf5(h5_path: str, camera_key: str = "cam_high", frame_idx: int = 0):
    """
    Extract a single frame from HDF5 file.

    Parameters:
    - h5_path: Path to HDF5 file
    - camera_key: Key for the camera in the HDF5 file (default: 'cam_high')
    - frame_idx: Index of frame to extract (default: 0 for first frame)

    Returns:
    - PIL Image or None if extraction fails
    """
    try:
        with h5py.File(h5_path, "r") as f:
            if "observations/images" not in f:
                print(f"Warning: 'observations/images' not found in {h5_path}")
                return None

            images_grp = f["observations/images"]

            if camera_key not in images_grp.keys():
                print(f"Warning: '{camera_key}' not found in {h5_path}. Available keys: {list(images_grp.keys())}")
                return None

            ds = images_grp[camera_key]  # (T, max_len)

            if frame_idx >= ds.shape[0]:
                print(f"Warning: frame_idx {frame_idx} >= num_frames {ds.shape[0]} in {h5_path}")
                return None

            raw = ds[frame_idx].tobytes().rstrip(b"\x00")
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            return img

    except Exception as e:
        print(f"Error extracting frame from {h5_path}: {e}")
        return None


def extract_video_from_hdf5(h5_path: str, camera_key: str = "cam_high"):
    """
    Extract all frames from HDF5 file as a list of PIL Images.

    Parameters:
    - h5_path: Path to HDF5 file
    - camera_key: Key for the camera in the HDF5 file (default: 'cam_high')

    Returns:
    - List of PIL Images or None if extraction fails
    """
    try:
        with h5py.File(h5_path, "r") as f:
            if "observations/images" not in f:
                print(f"Warning: 'observations/images' not found in {h5_path}")
                return None

            images_grp = f["observations/images"]

            if camera_key not in images_grp.keys():
                print(f"Warning: '{camera_key}' not found in {h5_path}. Available keys: {list(images_grp.keys())}")
                return None

            ds = images_grp[camera_key]  # (T, max_len)
            frames = []

            for t in range(ds.shape[0]):
                raw = ds[t].tobytes().rstrip(b"\x00")
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                frames.append(img)

            return frames

    except Exception as e:
        print(f"Error extracting video from {h5_path}: {e}")
        return None


def save_frames_as_png(episodes, data_dir, output_dir, camera_key, frame_idx=0):
    """
    Extract initial frames from HDF5 files and save as PNG.

    Parameters:
    - episodes: List of episode paths
    - data_dir: Root directory containing the data
    - output_dir: Directory to save PNG files
    - camera_key: Camera key to extract from
    - frame_idx: Frame index to extract (default: 0)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting initial frames (frame_idx={frame_idx}) from {len(episodes)} episodes...")

    for h5_path in tqdm.tqdm(episodes):
        # h5_path can be either a full path (from split file) or just filename
        if not os.path.isabs(h5_path):
            # Relative path, prepend data_dir
            h5_path = os.path.join(data_dir, h5_path)

        if not h5_path.endswith(".hdf5"):
            continue

        # Extract frame
        img = extract_frame_from_hdf5(h5_path, camera_key, frame_idx)

        if img is None:
            continue

        # Generate output filename
        base_name = Path(h5_path).stem  # filename without extension
        output_path = output_dir / f"{base_name}_frame{frame_idx}.png"

        # Save image
        img.save(output_path)

    print(f"✓ Saved frames to {output_dir}")


def save_video_as_mp4(episodes, data_dir, output_dir, camera_key, fps=30):
    """
    Extract videos from HDF5 files and save as MP4.

    Parameters:
    - episodes: List of episode paths
    - data_dir: Root directory containing the data
    - output_dir: Directory to save MP4 files
    - camera_key: Camera key to extract from
    - fps: Frames per second for output video (default: 30)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "ffmpeg is not installed or not in PATH. "
            "Please install ffmpeg: brew install ffmpeg (on macOS) or apt-get install ffmpeg (on Linux)"
        )

    print(f"Extracting videos from {len(episodes)} episodes...")

    for h5_path in tqdm.tqdm(episodes):
        # h5_path can be either a full path (from split file) or just filename
        if not os.path.isabs(h5_path):
            # Relative path, prepend data_dir
            h5_path = os.path.join(data_dir, h5_path)

        if not h5_path.endswith(".hdf5"):
            continue

        # Extract frames
        frames = extract_video_from_hdf5(h5_path, camera_key)

        if frames is None or len(frames) == 0:
            continue

        # Generate output filename
        base_name = Path(h5_path).stem  # filename without extension
        output_path = output_dir / f"{base_name}.mp4"

        # Save frames as temporary PNG files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            # Save frames as images
            for i, frame in enumerate(frames):
                frame_path = temp_dir / f"frame_{i:06d}.png"
                frame.save(frame_path)

            # Use ffmpeg to create video from frames
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-framerate",
                str(fps),
                "-i",
                str(temp_dir / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(fps),
                "-f",
                "mp4",
                str(output_path),
            ]

            # Run ffmpeg
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"⚠ Warning: ffmpeg failed for {h5_path}: {result.stderr}")
                continue

    print(f"✓ Saved videos to {output_dir}")


def main(
    data_dir: str,
    split_file: str,
    splits: list,
    output_dir: str,
    mode: str = "frame",
    camera_key: str = "cam_high",
    frame_idx: int = 0,
    fps: int = 30,
):
    """
    Main function to extract frames or videos from HDF5 files.

    Parameters:
    - data_dir: Root directory containing the data
    - split_file: Path to JSON file with train/test split
    - splits: List of split keys to use
    - output_dir: Directory to save output files
    - mode: 'frame' to extract initial frame, 'video' to extract full video
    - camera_key: Camera key to extract from (default: 'cam_high')
    - frame_idx: Frame index to extract when mode='frame' (default: 0)
    - fps: Frames per second for output video when mode='video' (default: 30)
    """
    # Load episodes from split file
    episodes = load_episodes_from_split(split_file, splits)
    print(f"Loaded {len(episodes)} episodes from splits: {', '.join(splits)}")

    if not episodes:
        raise ValueError("No episodes found in split file")

    # Process based on mode
    if mode == "frame":
        save_frames_as_png(episodes, data_dir, output_dir, camera_key, frame_idx)
    elif mode == "video":
        save_video_as_mp4(episodes, data_dir, output_dir, camera_key, fps)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'frame' or 'video'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames or videos from HDF5 files")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing the data",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        required=True,
        help="Path to JSON file with train/test split",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        required=True,
        help="Which splits to use (e.g., 'train_A', 'test_B')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["frame", "video"],
        default="frame",
        help="Mode: 'frame' to extract initial frame, 'video' to extract full video",
    )
    parser.add_argument(
        "--camera_key",
        type=str,
        default="cam_high",
        help="Camera key to extract from (default: 'cam_high')",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="Frame index to extract when mode='frame' (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for output video when mode='video' (default: 30)",
    )

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        split_file=args.split_file,
        splits=args.splits,
        output_dir=args.output_dir,
        mode=args.mode,
        camera_key=args.camera_key,
        frame_idx=args.frame_idx,
        fps=args.fps,
    )
