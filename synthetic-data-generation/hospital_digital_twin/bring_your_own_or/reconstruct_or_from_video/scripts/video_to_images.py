# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extract frames from video files at a specified FPS.

Simple script that processes all videos in a folder and extracts frames
at a target FPS rate, merging them into a single image set.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import cv2


def extract_frames_fps(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 2.0,
    start_index: int = 0,
) -> int:
    """
    Extract frames at a target FPS rate.

    Args:
        video_path: Path to input video
        output_dir: Output directory for frames
        target_fps: Target frames per second to extract
        start_index: Starting index for frame naming (for merging multiple videos)

    Returns:
        Number of frames extracted
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    print("Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  Video FPS: {video_fps:.2f}")
    print(f"  Duration: {total_frames/video_fps:.2f}s")
    print(f"  Target extraction FPS: {target_fps}")

    # Calculate frame interval based on FPS
    frame_interval = int(video_fps / target_fps)
    if frame_interval < 1:
        frame_interval = 1
        print(f"  Warning: Target FPS {target_fps} >= video FPS {video_fps:.2f}, extracting all frames")
    else:
        print(f"  Extracting every {frame_interval} frames...")

    if start_index > 0:
        print(f"  Starting from index {start_index} (continuing from previous video)")

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every Nth frame
        if frame_count % frame_interval == 0:
            output_path = output_dir / f"frame_{start_index + saved_count:06d}.jpg"
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

            if saved_count % 10 == 0:
                print(f"  Extracted {saved_count} frames...", end="\r")

        frame_count += 1

    cap.release()
    print(f"\nExtracted {saved_count} frames from {frame_count} total frames")
    return saved_count


def find_video_files(folder_path: Path) -> List[Path]:
    """
    Find all video files in a folder.

    Args:
        folder_path: Path to folder to search

    Returns:
        List of video file paths
    """
    video_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".flv",
        ".wmv",
        ".m4v",
        ".mpg",
        ".mpeg",
    }
    video_files = []

    for ext in video_extensions:
        video_files.extend(folder_path.glob(f"*{ext}"))
        video_files.extend(folder_path.glob(f"*{ext.upper()}"))

    return sorted(video_files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract frames from video files at a specified FPS.",
        epilog="Example: python video_to_images.py ./videos --fps 2",
    )
    parser.add_argument(
        "videos_folder",
        type=Path,
        help="Folder containing video file(s) to process",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        metavar="FPS",
        help="Target frames per second to extract (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory for frames (default: <videos_folder>/images)",
    )
    args = parser.parse_args()

    videos_folder = args.videos_folder.resolve()
    target_fps = args.fps
    output_dir = args.output.resolve() if args.output else videos_folder / "images"

    # Validate input folder
    if not videos_folder.exists():
        print(f"Error: Folder not found: {videos_folder}")
        return 1

    if not videos_folder.is_dir():
        print(f"Error: Not a directory: {videos_folder}")
        return 1

    # Find all videos
    video_files = find_video_files(videos_folder)
    if not video_files:
        print(f"Error: No video files found in {videos_folder}")
        return 1

    print(f"Found {len(video_files)} video file(s):")
    for vf in video_files:
        print(f"  - {vf.name}")

    print(f"\nImages will be saved to: {output_dir}")
    print(f"Target FPS: {target_fps}")

    # Process all videos, merging into one image set
    total_frames = 0
    frame_index = 0

    try:
        for i, video_file in enumerate(video_files, 1):
            print(f"\n{'=' * 70}")
            print(f"Processing video {i}/{len(video_files)}: {video_file.name}")
            print(f"{'=' * 70}")

            num_frames = extract_frames_fps(
                video_file,
                output_dir,
                target_fps=target_fps,
                start_index=frame_index,
            )

            frame_index += num_frames
            total_frames += num_frames

        print(f"\n{'=' * 70}")
        print(f"✓ Successfully extracted {total_frames} frames from {len(video_files)} video(s)")
        print(f"Output directory: {output_dir}")
        print(f"{'=' * 70}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
