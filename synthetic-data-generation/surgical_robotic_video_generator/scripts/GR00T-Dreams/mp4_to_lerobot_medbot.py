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
Convert a folder of MP4 files to LeRobot format for Medbot.
Creates a dataset with zero state and zero action, ready for IDM inference.
Uses the EXACT SAME preprocessing as training: resize_with_pad (maintains aspect ratio + padding).
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """
    Replicates tf.image.resize_with_pad for multiple images using PIL.
    Resizes images to target height and width without distortion by padding with zeros.

    THIS IS THE EXACT SAME FUNCTION USED IN TRAINING DATA CONVERSION.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> np.ndarray:
    """
    Replicates tf.image.resize_with_pad for one image using PIL.
    Resizes an image to target height and width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return np.array(image)  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return np.array(zero_image)


class MedbotFeatureDict:
    """Feature dictionary for Medbot with single video key."""

    def __init__(
        self,
        video_key: str = "left_endo",
        image_shape: tuple = (224, 224, 3),
        state_shape: tuple = (20,),
        action_shape: tuple = (20,),
    ):
        self.video_key = video_key
        self.image_key = f"observation.images.{video_key}"
        self.state_key = "observation.state"
        self.action_key = "action"
        self.image_shape = image_shape
        self.state_shape = state_shape
        self.action_shape = action_shape

    @property
    def features(self):
        """Return features dict for LeRobotDataset.create()"""
        return {
            self.image_key: {
                "dtype": "video",
                "shape": self.image_shape,
                "names": ["height", "width", "channels"],
            },
            self.state_key: {
                "dtype": "float32",
                "shape": self.state_shape,
                "names": ["state"],
            },
            self.action_key: {
                "dtype": "float32",
                "shape": self.action_shape,
                "names": ["action"],
            },
        }

    def __call__(self, image: np.ndarray, state: np.ndarray, action: np.ndarray) -> dict:
        """Create frame dictionary."""
        return {
            self.image_key: image,
            self.state_key: state,
            self.action_key: action,
        }


def process_mp4_file(video_path: Path, target_height: int = 224, target_width: int = 224):
    """
    Process an MP4 file and return frames using EXACT SAME preprocessing as training.

    Args:
        video_path: Path to MP4 file
        target_height: Target height (default: 224)
        target_width: Target width (default: 224)

    Returns:
        List of processed frames as numpy arrays
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply same preprocessing as training: resize_with_pad
        # This maintains aspect ratio and pads with zeros
        processed_frame = resize_with_pad(frame_rgb, target_height, target_width)

        frames.append(processed_frame)

    cap.release()
    return frames


def create_lerobot_dataset(
    video_dir: str,
    output_dir: str,
    task_description: str,
    video_key: str = "left_endo",
    fps: int = 16,
    prefix_dream: bool = False,
    robot_type: str = "medbot",
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
    exclude_ids: list = None,
):
    """
    Convert MP4 files to LeRobot format using official LeRobot API.
    Uses EXACT SAME preprocessing as training (resize_with_pad: maintains aspect ratio + padding).

    Args:
        video_dir: Directory containing MP4 files
        output_dir: Output directory for LeRobot dataset
        task_description: Task description for all videos
        video_key: Video key name (default: left_endo)
        fps: Target FPS (default: 16)
        prefix_dream: Whether to add <DREAM> prefix to task (default: False)
        robot_type: Robot type (default: medbot)
        image_writer_threads: Number of threads for image writing
        image_writer_processes: Number of processes for image writing
        exclude_ids: List of IDs to exclude (e.g., [820, 265]). Files with _ID_ in filename will be skipped.
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    # Get all MP4 files
    video_files = sorted(list(video_dir.glob("*.mp4")))
    if len(video_files) == 0:
        raise ValueError(f"No MP4 files found in {video_dir}")

    print(f"Found {len(video_files)} MP4 files")

    # Filter out excluded IDs
    if exclude_ids:
        filtered_files = []
        excluded_files = []

        for video_file in video_files:
            filename = video_file.name
            # Check if any exclude ID appears in the filename with underscores
            should_exclude = False
            for exclude_id in exclude_ids:
                if f"_{exclude_id}_" in filename:
                    should_exclude = True
                    excluded_files.append(filename)
                    break

            if not should_exclude:
                filtered_files.append(video_file)

        video_files = filtered_files
        print(f"Excluding {len(excluded_files)} files with IDs: {exclude_ids}")
        if excluded_files:
            print(
                f"  Excluded files: {', '.join(excluded_files[:5])}"
                + (f" ... and {len(excluded_files)-5} more" if len(excluded_files) > 5 else "")
            )
        print(f"Processing {len(video_files)} videos after exclusion")

    # Add <DREAM> prefix if requested
    if prefix_dream:
        full_task_description = f"<DREAM> {task_description}"
    else:
        full_task_description = task_description

    # Remove output directory if it exists
    if output_dir.exists():
        print(f"Removing existing dataset at {output_dir}")
        shutil.rmtree(output_dir)

    # Create feature builder
    feature_builder = MedbotFeatureDict(
        video_key=video_key,
        image_shape=(224, 224, 3),
        state_shape=(20,),
        action_shape=(20,),
    )

    # Create LeRobot dataset using official API
    print("\nCreating LeRobot dataset...")
    dataset = LeRobotDataset.create(
        repo_id=str(output_dir),
        robot_type=robot_type,
        fps=fps,
        features=feature_builder.features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    # Copy stats and modality from training dataset (CRITICAL for correct denormalization)
    print("Copying metadata from training dataset...")
    metadata_source = Path("IDM_dump/global_metadata/medbot")

    # Copy stats.json
    stats_source = metadata_source / "stats.json"
    stats_dest = output_dir / "meta" / "stats.json"

    if stats_source.exists():
        shutil.copy(stats_source, stats_dest)
        print(f"  ✓ Copied stats.json from {stats_source}")
    else:
        print(f"  ⚠ Warning: stats.json not found at {stats_source}")
        print("    Inference denormalization may be incorrect!")

    # Copy modality.json
    modality_source = metadata_source / "modality.json"
    modality_dest = output_dir / "meta" / "modality.json"

    if modality_source.exists():
        shutil.copy(modality_source, modality_dest)
        print(f"  ✓ Copied modality.json from {modality_source}")
    else:
        print(f"  ⚠ Warning: modality.json not found at {modality_source}")

    # Process each video
    total_frames = 0
    print(f"\nProcessing {len(video_files)} videos...")

    for video_file in tqdm(video_files, desc="Converting videos"):
        try:
            # Process MP4 file using SAME preprocessing as training
            frames = process_mp4_file(video_file, target_height=224, target_width=224)
            num_frames = len(frames)
            total_frames += num_frames

            # Create zero state and action
            zero_state = np.zeros(20, dtype=np.float32)
            zero_action = np.zeros(20, dtype=np.float32)

            # Add each frame to the dataset
            for frame in frames:
                frame_dict = feature_builder(
                    image=frame,
                    state=zero_state,
                    action=zero_action,
                )
                dataset.add_frame(frame_dict, task=full_task_description)

            # Save episode
            dataset.save_episode()

        except Exception as e:
            print(f"\nError processing {video_file.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Consolidate the dataset (important for finalizing)

    print("\n✅ Dataset created successfully!")
    print(f"   Output: {output_dir}")
    print(f"   Episodes: {len(video_files)}")
    print(f"   Total frames: {total_frames}")
    print(f"   Task: {full_task_description}")

    # Verify files were created
    print("\n📁 Dataset structure:")
    if (output_dir / "meta" / "info.json").exists():
        print("   ✓ meta/info.json")
    if (output_dir / "meta" / "tasks.jsonl").exists():
        print("   ✓ meta/tasks.jsonl")
    if (output_dir / "meta" / "episodes.jsonl").exists():
        print("   ✓ meta/episodes.jsonl")
    if (output_dir / "meta" / "stats.json").exists():
        print("   ✓ meta/stats.json (from training dataset)")
    else:
        print("   ⚠ meta/stats.json MISSING - denormalization will fail!")
    if (output_dir / "meta" / "modality.json").exists():
        print("   ✓ meta/modality.json (from training dataset)")
    else:
        print("   ⚠ meta/modality.json MISSING")
    data_files = list((output_dir / "data").glob("*.parquet")) if (output_dir / "data").exists() else []
    if data_files:
        print(f"   ✓ data/*.parquet ({len(data_files)} files)")
    video_dirs = list((output_dir / "videos").glob("*")) if (output_dir / "videos").exists() else []
    if video_dirs:
        print(f"   ✓ videos/ ({len(video_dirs)} subdirectories)")

    print("\nNext steps:")
    print("  1. Run IDM inference using idm_inference_simple.py")
    print("     PYTHONPATH=. python scripts/idm_inference_simple.py \\")
    print("         --checkpoint /path/to/checkpoint \\")
    print(f"         --dataset {output_dir} \\")
    print("         --output-dir ./inference_results \\")
    print("         --observation-indices 0 8")


def main():
    parser = argparse.ArgumentParser(description="Convert MP4 files to LeRobot format for Medbot")
    parser.add_argument("--video-dir", type=str, required=True, help="Directory containing MP4 files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--task-description", type=str, required=True, help="Task description for all videos")
    parser.add_argument("--video-key", type=str, default="left_endo", help="Video key name (default: left_endo)")
    parser.add_argument("--fps", type=int, default=16, help="Target FPS (default: 16)")
    parser.add_argument(
        "--add-dream-prefix",
        action="store_true",
        help="Add <DREAM> prefix to task description (for zero-state training)",
    )
    parser.add_argument("--robot-type", type=str, default="medbot", help="Robot type (default: medbot)")
    parser.add_argument(
        "--image-writer-threads", type=int, default=10, help="Number of threads for image writing (default: 10)"
    )
    parser.add_argument(
        "--image-writer-processes", type=int, default=5, help="Number of processes for image writing (default: 5)"
    )
    parser.add_argument(
        "--exclude-id",
        type=int,
        nargs="+",
        default=None,
        help="List of IDs to exclude. Files with _ID_ in filename will be skipped (e.g., --exclude-id 820 265 952)",
    )

    args = parser.parse_args()

    create_lerobot_dataset(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        task_description=args.task_description,
        video_key=args.video_key,
        fps=args.fps,
        prefix_dream=args.add_dream_prefix,
        robot_type=args.robot_type,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        exclude_ids=args.exclude_id,
    )


if __name__ == "__main__":
    main()
