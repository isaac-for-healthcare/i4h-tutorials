# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Script for converting suturing dataset to LeRobot format.

Usage:
python convert_dataset.py /path/to/suturing/data \
    [--repo_id REPO_ID] [--task_prompt TASK_PROMPT] [--image_shape IMAGE_SHAPE]

The script expects data in the format:
suturing_all/tissue_X/1_needle_pickup*/episode_timestamp/
where each episode contains:
- kinematics/ (zarr format with robot state and action data)
- endo_psm1/ (left endoscopic camera images)
- endo_psm2/ (right endoscopic camera images)
- left/ (left wrist camera images)
- right/ (right wrist camera images)

The resulting dataset will get saved to the $LEROBOT_HOME directory.
"""

import argparse
import io
import json
import os
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

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


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


class BaseFeatureDict:
    action_key: str = "action"
    left_endo_image_key: str = "observation.images.left_endo"
    right_endo_image_key: str = "observation.images.right_endo"
    left_wrist_image_key: str = "observation.images.left_wrist"
    right_wrist_image_key: str = "observation.images.right_wrist"
    state_key: str = "observation.state"

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (224, 224, 3),
        state_shape: tuple[int, ...] = (7,),
        actions_shape: tuple[int, ...] = (6,),
    ):
        self.image_shape = image_shape
        self.state_shape = state_shape
        self.actions_shape = actions_shape

    @property
    def features(self):
        features_dict = {
            self.left_endo_image_key: {
                "dtype": "video",
                "shape": self.image_shape,
                "names": ["height", "width", "channels"],
            },
            self.right_endo_image_key: {
                "dtype": "video",
                "shape": self.image_shape,
                "names": ["height", "width", "channels"],
            },
            self.left_wrist_image_key: {
                "dtype": "video",
                "shape": self.image_shape,
                "names": ["height", "width", "channels"],
            },
            self.right_wrist_image_key: {
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
                "shape": self.actions_shape,
                "names": ["action"],
            },
        }

        return features_dict

    def __call__(
        self, left_endo_img, right_endo_img, left_wrist_img, right_wrist_img, state, action, seg=None, depth=None
    ) -> dict:
        frame_data = {}
        img_h, img_w, _ = self.image_shape
        _ = self.features  # Access property to ensure it's evaluated

        # Assign mandatory fields
        frame_data[self.left_endo_image_key] = resize_with_pad(left_endo_img, img_h, img_w)
        frame_data[self.right_endo_image_key] = resize_with_pad(right_endo_img, img_h, img_w)
        frame_data[self.left_wrist_image_key] = resize_with_pad(left_wrist_img, img_h, img_w)
        frame_data[self.right_wrist_image_key] = resize_with_pad(right_wrist_img, img_h, img_w)
        frame_data[self.state_key] = state
        frame_data[self.action_key] = action
        return frame_data


def create_lerobot_dataset(
    output_path: str,
    features: dict,
    robot_type: str = "medbot",
    fps: int = 30,
    image_writer_threads: int = 10,
    image_writer_processes: int = 5,
):
    """
    Creates a LeRobot dataset with specified configurations.

    This function initializes a LeRobot dataset with the given parameters,
    defining the structure and features of the dataset.

    Parameters:
    - output_path: The path where the dataset will be saved.
    - features: A dictionary defining the features of the dataset.
    - robot_type: The type of robot.
    - fps: Frames per second for the dataset.
    - image_writer_threads: Number of threads for image writing.
    - image_writer_processes: Number of processes for image writing.

    Returns:
    - An instance of LeRobotDataset configured with the specified parameters.
    """

    if os.path.isdir(output_path):
        raise Exception(f"Output path {output_path} already exists.")

    return LeRobotDataset.create(
        repo_id=output_path,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


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


def main(
    data_dir: str,
    repo_id: str,
    task_prompt: str,
    feature_builder,
    split_file: str = None,
    splits: list = None,
    **dataset_config_kwargs,
):
    """
    Main function to convert suturing data to LeRobot format.

    This function processes suturing data in the specified directory structure,
    extracts relevant data from zarr files and images, and saves it in the LeRobot format.

    Parameters:
    - data_dir: Root directory containing the suturing data.
    - repo_id: Identifier for the dataset repository.
    - task_prompt: Description of the task for which the dataset is used.
    - split_file: Path to JSON file with train/test split (optional)
    - splits: List of split keys to combine (e.g., ['train1', 'train2'])
    - include_depth: Whether to include depth images in the dataset.
    - include_seg: Whether to include segmentation images in the dataset.
    - run_compute_stats: Whether to run compute stats.
    - dataset_config_kwargs: Additional keyword arguments for dataset configuration.
    - feature_builder: An instance of a feature dictionary builder class (e.g., GR00TN1FeatureDict).
    """
    final_output_path = Path(repo_id)
    if final_output_path.exists():
        try:
            shutil.rmtree(final_output_path)
        except Exception as e:
            raise Exception(f"Error removing {final_output_path}: {e}. Please ensure that you have write permissions.")

    robot_type = dataset_config_kwargs.pop("robot_type", "medbot")
    fps = dataset_config_kwargs.pop("fps", 16)
    image_writer_threads = dataset_config_kwargs.pop("image_writer_threads", 10)
    image_writer_processes = dataset_config_kwargs.pop("image_writer_processes", 5)

    dataset = create_lerobot_dataset(
        output_path=final_output_path,
        features=feature_builder.features,
        robot_type=robot_type,
        fps=fps,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    # Find all suturing episodes
    if split_file and splits:
        # Load from split file
        episodes = load_episodes_from_split(split_file, splits)
        print(f"Loaded {len(episodes)} episodes from splits: {', '.join(splits)}")
    else:
        # Original behavior: use all files from data_dir (excluding last 5)
        all_files = [f for f in os.listdir(data_dir) if f.endswith(".hdf5")]
        episodes = [os.path.join(data_dir, f) for f in all_files]
        print(f"Found {len(episodes)} episodes to process (original mode)")

    if not episodes:
        warnings.warn("No episodes found")
        return

    print(f"Processing {len(episodes)} episodes")

    # Process each episode
    for h5_path in tqdm.tqdm(episodes):
        # h5_path can be either a full path (from split file) or just filename
        if not os.path.isabs(h5_path):
            # Relative path, prepend data_dir
            h5_path = os.path.join(data_dir, h5_path)

        if not h5_path.endswith(".hdf5"):
            continue

        with h5py.File(h5_path, "r") as f:
            # Actions
            action = f["action"][()]
            state = action.copy()
            # Save images frame by frame
            left_endo_images, right_endo_images, left_wrist_images, right_wrist_images = [], [], [], []
            images_grp = f["observations/images"]
            for t in range(action.shape[0]):
                img = []
                if "cam_lascope" not in images_grp.keys():
                    # the new collected data does not have cam_lascope image
                    cam_names = ["cam_high", "cam_high", "cam_left_wrist", "cam_right_wrist"]
                else:
                    cam_names = ["cam_high", "cam_lascope", "cam_left_wrist", "cam_right_wrist"]
                for cam in cam_names:
                    ds = images_grp[cam]  # (T, max_len)
                    raw = ds[t].tobytes().rstrip(b"\x00")
                    _img = Image.open(io.BytesIO(raw)).convert("RGB")
                    img.append(np.array(_img))
                left_endo_images.append(img[0])
                right_endo_images.append(img[1])
                left_wrist_images.append(img[2])
                right_wrist_images.append(img[3])

        # Check that all sequences have the same length
        num_steps = action.shape[0]
        for step in range(num_steps):
            # Create frame dictionary
            frame_dict = feature_builder(
                left_endo_img=left_endo_images[step],
                right_endo_img=right_endo_images[step],
                left_wrist_img=left_wrist_images[step],
                right_wrist_img=right_wrist_images[step],
                state=state[step],
                action=action[step],
            )
            # Add task to the frame
            if os.path.exists(task_prompt):
                # task prompt is a folder, read the text file inside
                prompt_file = os.path.join(task_prompt, f"{os.path.basename(h5_path).replace('.hdf5', '.txt')}")
                try:
                    with open(prompt_file, "r") as pf:
                        episode_task = pf.read().strip()
                except OSError:
                    episode_task = (
                        "phantom: Left robotic forcep failed to pick up the needle and "
                        "grabbed the rubber pad. Right forcep did nothing"
                    )
            else:
                episode_task = f"{task_prompt}"
            dataset.add_frame(frame_dict, task=episode_task)

        # Save episode
        dataset.save_episode()

    print(f"Saving dataset to {final_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 files to LeRobot format")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="medbot_lerobot",
        help="Directory to save the dataset under (relative to LEROBOT_HOME)",
    )
    parser.add_argument(
        "--task_prompt",
        type=str,
        default=(
            "The left arm of the surgical robot is picking up a needle over a red rubber pad "
            "and handing it over to the right arm."
        ),
        help="Prompt description of the task",
    )
    parser.add_argument(
        "--image_shape",
        type=lambda s: tuple(map(int, s.split(","))),
        default=(224, 224, 3),
        help="Shape of the image data as a comma-separated string, e.g., '224,224,3'",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Path to JSON file with train/test split (optional)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Which splits to use (can combine multiple): 'train1', 'train2', 'test1', 'test2'. "
        "Examples: --splits train1 train2 (combine both), --splits test1 (test only from folder A)",
    )

    args = parser.parse_args()

    # Validate split arguments
    if args.split_file and not args.splits:
        parser.error("--splits is required when --split_file is provided")
    if args.splits and not args.split_file:
        parser.error("--split_file is required when --splits is provided")

    # Instantiate the feature builder based on args
    # For suturing data: state has 14 dimensions (12 joint states + 2 jaw positions)
    # Action has 16 dimensions (PSM1 pose: 3+4, PSM2 pose: 3+4, jaw positions: 2)
    feature_builder = BaseFeatureDict(
        image_shape=args.image_shape,
        state_shape=(20,),  # 12 joint states + 2 jaw positions
        actions_shape=(20,),  # PSM1 pose (7) + PSM2 pose (7) + jaw positions (2)
    )
    main(
        args.data_dir,
        args.repo_id,
        args.task_prompt,
        feature_builder=feature_builder,
        split_file=args.split_file,
        splits=args.splits,
    )
