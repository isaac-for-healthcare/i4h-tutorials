#!/usr/bin/env bash
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
  echo "Loaded paths from $ENV_FILE"
else
  echo "Error: $ENV_FILE not found. Copy .env.template and fill in the paths." >&2
  exit 1
fi

# Validate required variables used in volume mounts
MISSING=()
for VAR in COSMOS_HF_HOME COSMOS_REPO_DIR COSMOS_ROBOTIC_US_DIR COSMOS_VENV_DIR COSMOS_UV_CACHE_DIR; do
  [ -z "${!VAR}" ] && MISSING+=("$VAR")
done
if [ "${#MISSING[@]}" -gt 0 ]; then
  echo "Error: The following required variables are unset in $ENV_FILE: ${MISSING[*]}" >&2
  exit 1
fi

# Determine NUM_GPUS, allowing override via environment and falling back to detection
if [ -z "${NUM_GPUS:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -le 0 ] 2>/dev/null; then
      NUM_GPUS=1
    fi
  else
    NUM_GPUS=1
  fi
fi

# Determine OMP_NUM_THREADS, allowing override via environment and defaulting to threads per GPU
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  TOTAL_CPUS=$(nproc)
  if [ -z "$TOTAL_CPUS" ] || [ "$TOTAL_CPUS" -le 0 ] 2>/dev/null; then
    OMP_NUM_THREADS=1
  else
    if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -le 0 ] 2>/dev/null; then
      OMP_NUM_THREADS=$TOTAL_CPUS
    else
      THREADS_PER_GPU=$(( TOTAL_CPUS / NUM_GPUS ))
      if [ "$THREADS_PER_GPU" -le 0 ]; then
        THREADS_PER_GPU=1
      fi
      OMP_NUM_THREADS=$THREADS_PER_GPU
    fi
  fi
fi

# Volume mounts: workspace, cache, venv, tutorials, and host identity
VOL="-v $COSMOS_HF_HOME:/workspace/.cache/huggingface"
# Repository directory for cosmos-transfer2.5
VOL="$VOL -v $COSMOS_REPO_DIR:/workspace"
# i4h-tutorials repository
VOL="$VOL -v $SCRIPT_DIR/../..:/workspace/i4h-tutorials"
# Robotic ultrasound data directory
VOL="$VOL -v $COSMOS_ROBOTIC_US_DIR:/workspace/robotic_us_example"
VOL="$VOL -v $COSMOS_VENV_DIR:/workspace/.venv"
VOL="$VOL -v $COSMOS_UV_CACHE_DIR:/workspace/.cache/uv"
# Use real resolv.conf (avoids systemd-resolved stub at 127.0.0.53)
[ -f /run/systemd/resolve/resolv.conf ] && VOL="$VOL -v /run/systemd/resolve/resolv.conf:/etc/resolv.conf:ro"
VOL="$VOL -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro"

# Container name (can be overridden via GUIDED_TRANSFER_CONTAINER_NAME in .env)
CONTAINER_NAME="${GUIDED_TRANSFER_CONTAINER_NAME:-guided-transfer2.5}"

docker run -it --rm --runtime=nvidia --network=host --ipc=host --user "$(id -u):$(id -g)" --name "$CONTAINER_NAME" \
  $VOL \
  -e HOME=/workspace \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e OMP_NUM_THREADS=$OMP_NUM_THREADS \
  -e NUM_GPUS=$NUM_GPUS \
  -e HF_HUB_DISABLE_XET=1 \
  -e UV_PYTHON_DOWNLOADS=never \
  ${DATA_DIR:+-e DATA_DIR="$DATA_DIR"} \
  guided-transfer2.5
