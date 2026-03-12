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
Convert HDF5 files to MP4 videos and NPZ segmentation masks.

This script processes HDF5 files containing robot teleoperation data and converts
the image observations (RGB, depth, segmentation) into MP4 videos for visualization.
Additionally, it saves the raw segmentation masks as compressed NPZ files for further processing.
Uses ffmpeg for MP4 conversion to ensure compatibility.

Outputs per episode:
- RGB videos (one per camera): rgb_video_0.mp4, rgb_video_1.mp4, ...
- Depth videos (one per camera): depth_video_0.mp4, depth_video_1.mp4, ...
- Segmentation videos (one per camera): seg_mask_video_0.mp4, seg_mask_video_1.mp4, ...
- RGB images (NPZ): rgb_images.npz - compressed array with shape (n_frames, n_cameras, height, width, 3)
- Depth images (NPZ): depth_images.npz - compressed array with shape (n_frames, n_cameras, height, width, 1)
- Segmentation masks (NPZ): seg_masks.npz - compressed array with shape (n_frames, n_cameras, height, width, 3)
- Camera parameters (NPZ): room_camera_para.npz, wrist_camera_para.npz

Usage:
    python hdf5_to_video.py \\
        --input_dir /path/to/hdf5/folder \\
        --output_dir ./videos
"""

import argparse
import glob
import os
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np


def convert_hdf5_to_videos(h5_file_path: str, output_dir: str, episode_name: str, upscale_720p: bool = False) -> dict:
    """
    Convert a single HDF5 file to MP4 videos and NPZ masks.

    Args:
        h5_file_path: Path to the HDF5 file
        output_dir: Directory to save the videos
        episode_name: Name for the episode subfolder
        upscale_720p: If True, upscale videos to 720p (1280x720) resolution

    Returns:
        Dictionary with paths to the created videos
    """
    print(f"\nProcessing: {h5_file_path}")
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_file_path, "r") as f:
        # Load image data
        rgb_images = f["data/demo_0/observations/rgb_images"][:]  # (n_frames, 2, 224, 224, 3)
        depth_images = f["data/demo_0/observations/depth_images"][:]  # (n_frames, 2, 224, 224, 1)
        seg_images = f["data/demo_0/observations/seg_images"][:]  # (n_frames, 2, 224, 224, 3)

        # Video parameters
        num_frames, num_videos, height, width, _ = seg_images.shape
        fps = 30  # Frames per second
        # Use XVID codec with AVI format for temporary files
        video_codec = cv2.VideoWriter_fourcc(*"XVID")

        print(f"Frames: {num_frames}, Cameras: {num_videos}, Resolution: {width}x{height}, FPS: {fps}")

        # Create temporary AVI paths and final MP4 paths
        temp_video_paths = {
            "rgb": [os.path.join(output_dir, f"rgb_video_{i}_temp.avi") for i in range(num_videos)],
            "depth": [os.path.join(output_dir, f"depth_video_{i}_temp.avi") for i in range(num_videos)],
            "seg": [os.path.join(output_dir, f"seg_mask_video_{i}_temp.avi") for i in range(num_videos)],
        }

        video_paths = {
            "rgb": [os.path.join(output_dir, f"rgb_video_{i}.mp4") for i in range(num_videos)],
            "depth": [os.path.join(output_dir, f"depth_video_{i}.mp4") for i in range(num_videos)],
            "seg": [os.path.join(output_dir, f"seg_mask_video_{i}.mp4") for i in range(num_videos)],
        }

        writers = {
            "rgb": [cv2.VideoWriter(path, video_codec, fps, (width, height)) for path in temp_video_paths["rgb"]],
            "depth": [
                cv2.VideoWriter(path, video_codec, fps, (width, height), isColor=False)
                for path in temp_video_paths["depth"]
            ],
            "seg": [
                cv2.VideoWriter(path, video_codec, fps, (width, height), isColor=True)
                for path in temp_video_paths["seg"]
            ],
        }

        # Write frames
        for i in range(num_frames):
            for vid_idx in range(num_videos):
                # RGB Video
                rgb_frame = rgb_images[i, vid_idx].astype(np.uint8)[:, :, ::-1]  # BGR for OpenCV
                writers["rgb"][vid_idx].write(rgb_frame)

                # Depth Video - normalize and convert
                depth_frame = depth_images[i, vid_idx, :, :, 0]
                output = 1.0 / (depth_frame + 1e-6)
                depth_min = output.min()
                depth_max = output.max()
                max_val = 255  # Maximum value for uint8

                if depth_max - depth_min > np.finfo("float").eps:
                    out_array = max_val * (output - depth_min) / (depth_max - depth_min)
                else:
                    out_array = np.zeros_like(output)

                formatted = out_array.astype("uint8")
                writers["depth"][vid_idx].write(formatted)

                # Segmentation Video
                seg_mask_frame = seg_images[i, vid_idx, :, :, :].astype(np.uint8)
                writers["seg"][vid_idx].write(seg_mask_frame)

        # Release all video writers
        for category in writers:
            for writer in writers[category]:
                writer.release()

        # Convert AVI files to MP4 using ffmpeg
        if upscale_720p:
            print("Converting videos to MP4 format and upscaling to 720p using ffmpeg...")
        else:
            print("Converting videos to MP4 format using ffmpeg...")

        for category in ["rgb", "depth", "seg"]:
            for i, (temp_path, final_path) in enumerate(zip(temp_video_paths[category], video_paths[category])):
                try:
                    # Build ffmpeg command
                    ffmpeg_cmd = ["ffmpeg", "-i", temp_path]

                    if upscale_720p:
                        # Use different scaling algorithms based on content type
                        if category == "seg":
                            # Use nearest neighbor for segmentation masks to preserve class labels
                            ffmpeg_cmd.extend(["-vf", "scale=1280:720:flags=neighbor"])
                        else:
                            # Use lanczos (high quality) for RGB and depth
                            ffmpeg_cmd.extend(["-vf", "scale=1280:720:flags=lanczos"])

                    ffmpeg_cmd.extend(
                        [
                            "-c:v",
                            "libx264",
                            "-preset",
                            "medium",
                            "-crf",
                            "23",
                            "-y",  # Overwrite output file if exists
                            final_path,
                        ]
                    )

                    subprocess.run(
                        ffmpeg_cmd,
                        check=True,
                        capture_output=True,
                    )
                    # Remove temporary AVI file
                    os.remove(temp_path)
                except subprocess.CalledProcessError as e:
                    print(f"Warning: ffmpeg conversion failed for {temp_path}: {e.stderr.decode()}")
                    # Keep the temp file if conversion fails
                except Exception as e:
                    print(f"Warning: Error converting {temp_path}: {e}")

        # Save RGB, depth, and segmentation data as NPZ for further processing
        rgb_npz_path = os.path.join(output_dir, "rgb_images.npz")
        depth_npz_path = os.path.join(output_dir, "depth_images.npz")
        seg_masks_path = os.path.join(output_dir, "seg_masks.npz")

        np.savez_compressed(
            rgb_npz_path,
            rgb_images=rgb_images,  # (n_frames, n_cameras, height, width, 3)
        )
        np.savez_compressed(
            depth_npz_path,
            depth_images=depth_images,  # (n_frames, n_cameras, height, width, 1)
        )
        np.savez_compressed(
            seg_masks_path,
            seg_images=seg_images,  # (n_frames, n_cameras, height, width, 3)
        )
        print(f"✓ Saved RGB images to: {rgb_npz_path}")
        print(f"✓ Saved depth images to: {depth_npz_path}")
        print(f"✓ Saved segmentation masks to: {seg_masks_path}")

        # Save camera parameters and NPZ paths
        npz_data = {
            "rgb_npz_path": rgb_npz_path,
            "depth_npz_path": depth_npz_path,
            "seg_masks_path": seg_masks_path,
        }
        camera_params = npz_data.copy()

        # Calculate scaling factors for camera intrinsics if upscaling
        if upscale_720p:
            # Original resolution is 224x224, upscaling to 1280x720
            width_scale = 1280.0 / width
            height_scale = 720.0 / height
            print(f"Scaling camera intrinsics: width_scale={width_scale:.3f}, height_scale={height_scale:.3f}")

        # Room camera parameters
        if "data/demo_0/observations/room_camera_intrinsic_matrices" in f:
            room_camera_intrinsic_matrices = f["data/demo_0/observations/room_camera_intrinsic_matrices"][:].copy()
            room_camera_pos = f["data/demo_0/observations/room_camera_pos"][:]
            room_camera_quat = f["data/demo_0/observations/room_camera_quat_w_ros"][:]

            # Scale intrinsic matrices if upscaling
            if upscale_720p:
                # Intrinsic matrix format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                # Scale fx, fy (focal lengths) and cx, cy (principal points)
                for i in range(room_camera_intrinsic_matrices.shape[0]):  # For each frame
                    room_camera_intrinsic_matrices[i, 0, 0] *= width_scale  # fx
                    room_camera_intrinsic_matrices[i, 1, 1] *= height_scale  # fy
                    room_camera_intrinsic_matrices[i, 0, 2] *= width_scale  # cx
                    room_camera_intrinsic_matrices[i, 1, 2] *= height_scale  # cy

            room_camera_para_path = os.path.join(output_dir, "room_camera_para.npz")
            np.savez(
                room_camera_para_path,
                room_camera_intrinsic_matrices=room_camera_intrinsic_matrices,
                room_camera_pos=room_camera_pos,
                room_camera_quat=room_camera_quat,
            )
            camera_params["room_camera_para_path"] = room_camera_para_path

        # Wrist camera parameters
        if "data/demo_0/observations/wrist_camera_intrinsic_matrices" in f:
            wrist_camera_intrinsic_matrices = f["data/demo_0/observations/wrist_camera_intrinsic_matrices"][:].copy()
            wrist_camera_pos = f["data/demo_0/observations/wrist_camera_pos"][:]
            wrist_camera_quat = f["data/demo_0/observations/wrist_camera_quat_w_ros"][:]

            # Scale intrinsic matrices if upscaling
            if upscale_720p:
                # Intrinsic matrix format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                # Scale fx, fy (focal lengths) and cx, cy (principal points)
                for i in range(wrist_camera_intrinsic_matrices.shape[0]):  # For each frame
                    wrist_camera_intrinsic_matrices[i, 0, 0] *= width_scale  # fx
                    wrist_camera_intrinsic_matrices[i, 1, 1] *= height_scale  # fy
                    wrist_camera_intrinsic_matrices[i, 0, 2] *= width_scale  # cx
                    wrist_camera_intrinsic_matrices[i, 1, 2] *= height_scale  # cy

            wrist_camera_para_path = os.path.join(output_dir, "wrist_camera_para.npz")
            np.savez(
                wrist_camera_para_path,
                wrist_camera_intrinsic_matrices=wrist_camera_intrinsic_matrices,
                wrist_camera_pos=wrist_camera_pos,
                wrist_camera_quat=wrist_camera_quat,
            )
            camera_params["wrist_camera_para_path"] = wrist_camera_para_path

    result = {
        "episode_name": episode_name,
        "output_dir": output_dir,
        "num_frames": num_frames,
        "num_cameras": num_videos,
        **video_paths,
        **camera_params,
    }

    print(f"✓ Created {num_videos * 3} videos in: {output_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 teleoperation data to MP4 videos and NPZ masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all HDF5 files in a folder (saves to {input_dir}/videos by default)
  python hdf5_to_video.py \\
      --input_dir /media/data/data/hdf5/2026-01-30-15-54-Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0

  # Convert and upscale to 720p (as used in Cosmos Transfer 2.5 training)
  python hdf5_to_video.py \\
      --input_dir /path/to/hdf5/folder \\
      --output_dir ./videos \\
      --upscale-720p

  # Process a single HDF5 file with custom output directory
  python hdf5_to_video.py \\
      --input_dir /path/to/folder \\
      --output_dir /custom/output/path \\
      --pattern "data_0.hdf5"
        """,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing HDF5 files (e.g., data_0.hdf5, data_1.hdf5, ...)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output videos (default: {input_dir}/videos)",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.hdf5",
        help="File pattern to match (default: *.hdf5)",
    )

    parser.add_argument(
        "--upscale-720p",
        action="store_true",
        help="Upscale videos to 720p (1280x720) resolution. Uses lanczos for RGB/depth, nearest neighbor for masks.",
    )

    args = parser.parse_args()

    # Set default output directory if not provided
    if args.output_dir is None:
        input_path = Path(args.input_dir)
        args.output_dir = str(input_path / "videos")

    # Find all HDF5 files
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return

    h5_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))

    if not h5_files:
        print(f"Error: No HDF5 files found in {args.input_dir} matching pattern '{args.pattern}'")
        return

    print(f"Found {len(h5_files)} HDF5 file(s) to process")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Process each HDF5 file
    results = []
    for i, h5_file in enumerate(h5_files):
        episode_name = Path(h5_file).stem  # e.g., "data_0" from "data_0.hdf5"
        episode_output_dir = os.path.join(args.output_dir, episode_name)

        try:
            result = convert_hdf5_to_videos(h5_file, episode_output_dir, episode_name, args.upscale_720p)
            results.append(result)
        except Exception as e:
            print(f"✗ Error processing {h5_file}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print(f"Conversion complete! Processed {len(results)}/{len(h5_files)} files")
    print(f"\nVideos saved to: {args.output_dir}")

    # Print summary
    print("\nSummary:")
    for result in results:
        print(f"  - {result['episode_name']}: {result['num_frames']} frames, {result['num_cameras']} cameras")


if __name__ == "__main__":
    main()
