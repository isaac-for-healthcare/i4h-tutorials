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

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import List


def run_docker(step_name: str, docker_args: List[str], cleanup_on_success: bool = True) -> None:
    """
    Run a docker container with real-time output streaming.

    Args:
        step_name: Human-readable step name for the container
        docker_args: Full docker run command arguments
        cleanup_on_success: Whether to remove container on success
    """
    # Extract container name from docker_args (should be after --name flag)
    container_name = None
    for i, arg in enumerate(docker_args):
        if arg == "--name" and i + 1 < len(docker_args):
            container_name = docker_args[i + 1]
            break

    pretty = " ".join([shlex.quote(x) for x in docker_args])
    print(f"\n[{step_name}] Running docker command:")
    print(f"  $ {pretty}\n")

    try:
        # Use subprocess.run with stdout/stderr inherited for real-time output
        subprocess.run(docker_args, check=True, stdout=None, stderr=None)
        print(f"\n[{step_name}] ✓ Completed successfully")

        # Cleanup on success
        if cleanup_on_success and container_name:
            print(f"[{step_name}] Cleaning up container: {container_name}")
            subprocess.run(
                ["docker", "rm", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except subprocess.CalledProcessError as e:
        print(f"\n[{step_name}] ✗ Failed with exit code {e.returncode}")
        if container_name:
            print(f"\n{'='*70}")
            print(f"ERROR: {step_name} failed!")
            print(f"{'='*70}")
            print(f"Container '{container_name}' has been kept for debugging.\n")
            print("View the full logs:")
            print(f"  docker logs {container_name}\n")
            print("Save logs to a file:")
            print(f"  docker logs {container_name} > error.log 2>&1\n")
            print("Remove the container when done:")
            print(f"  docker rm {container_name}")
            print(f"{'='*70}\n")
        raise


def ensure_colmap_images_layout(work_dir: Path) -> None:
    """
    Enforce a COLMAP dataset layout compatible with 3dgrut:

      work_dir/
        images/                  # user inputs
        colmap/
          images -> ../images    # symlink (created automatically)
          sparse/0/...           # COLMAP output

    3dgrut loads images from <path>/images, where <path> is the COLMAP root.
    """
    images_src = work_dir / "images"
    if not images_src.is_dir():
        raise SystemExit(f"Missing images dir: {images_src}")

    colmap_dir = work_dir / "colmap"
    images_dst = colmap_dir / "images"

    colmap_dir.mkdir(parents=True, exist_ok=True)

    # Only create symlink if it doesn't already exist
    if not images_dst.exists():
        # IMPORTANT: use a *relative* symlink so it works inside docker too.
        images_dst.symlink_to(Path("..") / "images", target_is_directory=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="NuRec Pipeline")
    ap.add_argument("--work-dir", required=True, help="Working directory (must contain images/)")
    ap.add_argument(
        "--out-dir",
        default="out",
        help="Relative folder inside work-dir for 3dgrut outputs",
    )

    # 3dgrut knobs
    ap.add_argument("--experiment-name", default="3dgut_mcmc", help="3dgrut experiment name")
    ap.add_argument(
        "--config-name",
        default="apps/colmap_3dgut_mcmc.yaml",
        help="Hydra config for 3dgrut",
    )
    ap.add_argument("--skip-train", action="store_true", help="Skip 3dgrut training/export")

    # COLMAP knobs (from NVIDIA mono workflow docs)
    ap.add_argument(
        "--camera-model",
        default="PINHOLE",
        help="COLMAP camera model (PINHOLE or SIMPLE_PINHOLE)",
    )
    ap.add_argument(
        "--single-camera",
        type=int,
        default=1,
        help="COLMAP ImageReader.single_camera (1 recommended)",
    )
    ap.add_argument(
        "--max-image-size",
        type=int,
        default=2000,
        help="COLMAP SiftExtraction.max_image_size",
    )
    ap.add_argument("--use-gpu-matching", type=int, default=1, help="COLMAP SiftMatching.use_gpu")

    args = ap.parse_args()

    work_dir = Path(args.work_dir).expanduser().resolve()
    if not work_dir.is_dir():
        raise SystemExit(f"work-dir not found: {work_dir}")

    # Infer repo root from this file location:
    #   NuRec/scripts/reconstruct.py -> NuRec/
    repo_root = Path(__file__).resolve().parents[1]

    # 3dgrut should be cloned inside NuRec folder
    grut_repo = repo_root / "3dgrut"
    grut_dockerfile = grut_repo / "Dockerfile"

    if not grut_dockerfile.is_file():
        raise SystemExit(f"3dgrut Dockerfile not found at {grut_dockerfile}\n" f"Please clone 3dgrut repository.")

    # 1) Run COLMAP to generate sparse reconstruction
    colmap_root_rel = "colmap"
    colmap_root = work_dir / colmap_root_rel
    colmap_sparse0 = colmap_root / "sparse" / "0"

    colmap_output_exists = colmap_sparse0.is_dir()
    if colmap_output_exists:
        print(f"[pipeline] COLMAP output already exists at {colmap_sparse0}; skipping COLMAP.")

    if not colmap_output_exists:
        # Ensure and create the colmap-compatible dataset layout
        ensure_colmap_images_layout(work_dir)

        # Ensure output folder exists on host (so it persists after container exits)
        colmap_root.mkdir(parents=True, exist_ok=True)

        working = "/working"
        colmap_dataset = f"{working}/{colmap_root_rel}"

        # COLMAP pipeline
        # 1. Feature extraction
        # 2. Feature matching
        # 3. Global SFM
        # Refer to: https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html
        cmd = (
            "set -euo pipefail; "
            "mkdir -p {colmap_dataset}/sparse; "
            "echo '[COLMAP] Step 1/3: Feature extraction...'; "
            "colmap feature_extractor "
            "  --database_path {colmap_dataset}/database.db "
            "  --image_path {colmap_dataset}/images "
            "  --ImageReader.single_camera {single_camera} "
            "  --ImageReader.camera_model {camera_model} "
            "  --SiftExtraction.max_image_size {max_image_size} "
            "  --SiftExtraction.estimate_affine_shape 1 "
            "  --SiftExtraction.domain_size_pooling 1; "
            "echo '[COLMAP] Step 2/3: Feature matching...'; "
            "colmap exhaustive_matcher "
            "  --database_path {colmap_dataset}/database.db "
            "  --FeatureMatching.use_gpu {use_gpu_matching}; "
            "echo '[COLMAP] Step 3/3: Mapper (SfM)...'; "
            "colmap mapper "
            "  --database_path {colmap_dataset}/database.db "
            "  --image_path {colmap_dataset}/images "
            "  --output_path {colmap_dataset}/sparse; "
            "echo '[COLMAP] All steps completed successfully!'"
        ).format(
            colmap_dataset=colmap_dataset,
            single_camera=args.single_camera,
            camera_model=args.camera_model,
            max_image_size=args.max_image_size,
            use_gpu_matching=args.use_gpu_matching,
        )

        run_docker(
            "COLMAP Pipeline",
            [
                "docker",
                "run",
                "--name",
                "colmap-pipeline",
                "--gpus",
                "all",
                "--runtime=nvidia",
                "-v",
                f"{work_dir}:{working}",
                "-w",
                working,
                "colmap/colmap:20251107.4118",
                "bash",
                "-lc",
                cmd,
            ],
        )

    # Stop here if skip-train is set
    if args.skip_train:
        print("[pipeline] skip-train set; done after COLMAP.")
        return 0

    # 2) Train Dense 3D Reconstruction with 3DGUT

    # Build 3dgrut env image using the Dockerfile from the cloned 3dgrut repo
    image_tag = "3dgrut"
    subprocess.run(
        [
            "docker",
            "build",
            "-f",
            str(grut_dockerfile),
            "-t",
            image_tag,
            str(grut_repo),
        ],
        check=True,
    )

    # Run 3dgrut training in container
    # Refer to 3dgrut repo for more parameters
    data_root = "/data"
    train_cmd = [
        "conda",
        "run",
        "--no-capture-output",  # Prevent conda from buffering output
        "-n",
        "3dgrut",
        "python",
        "-u",  # Unbuffered Python output
        "train.py",
        "--config-name",
        args.config_name,
        f"path={data_root}/{colmap_root_rel}",
        f"out_dir={data_root}/{args.out_dir.strip('/').strip()}",
        f"experiment_name={args.experiment_name}",
        "export_usdz.enabled=true",
        "export_usdz.apply_normalizing_transform=true",
    ]

    run_docker(
        "3DGUT Training",
        [
            "docker",
            "run",
            "--name",
            "3dgrut-training",
            "-it",
            "--gpus",
            "all",
            "--runtime=nvidia",
            "--ipc=host",
            "-v",
            f"{work_dir}:{data_root}",
            image_tag,
            *train_cmd,
        ],
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
