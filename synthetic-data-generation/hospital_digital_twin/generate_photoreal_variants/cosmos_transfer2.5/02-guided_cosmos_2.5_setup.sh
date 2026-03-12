#!/bin/bash

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

# Detect run user (Brev/cloud defaults or current user).
if id "nvidia" &>/dev/null; then
  RUN_USER="nvidia"
elif id "shadeform" &>/dev/null; then
  RUN_USER="shadeform"
elif id "ubuntu" &>/dev/null; then
  RUN_USER="ubuntu"
elif [ -n "$SUDO_USER" ]; then
  RUN_USER="$SUDO_USER"
else
  RUN_USER="$(whoami)"
fi
RUN_GROUP="$(id -gn "$RUN_USER" 2>/dev/null)" || RUN_GROUP="$RUN_USER"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
I4H_TUTORIALS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load .env so COSMOS_* are set before we bake them into the inner script
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.env"
  set +a
  echo "Loaded paths from $SCRIPT_DIR/.env"
fi

# Write inner script to temp file (avoids nested-quote syntax errors; SCRIPT_DIR and I4H_TUTORIALS_ROOT baked in)
INNER_SCRIPT="/tmp/02_guided_cosmos_setup_inner_$$.sh"
cat << INNEREOF > "$INNER_SCRIPT"
#!/bin/bash
set -e
SCRIPT_DIR='$SCRIPT_DIR'
I4H_TUTORIALS_ROOT='$I4H_TUTORIALS_ROOT'
COSMOS_REPO_DIR='${COSMOS_REPO_DIR:-}'
COSMOS_HF_HOME='${COSMOS_HF_HOME:-}'
COSMOS_VENV_DIR='${COSMOS_VENV_DIR:-}'
COSMOS_TORCH_HOME='${COSMOS_TORCH_HOME:-}'
COSMOS_PIP_CACHE_DIR='${COSMOS_PIP_CACHE_DIR:-}'
COSMOS_ROBOTIC_US_DIR='${COSMOS_ROBOTIC_US_DIR:-}'
COSMOS_UV_CACHE_DIR='${COSMOS_UV_CACHE_DIR:-}'

section() { echo ""; echo "=== Section \$1: \$2 ==="; }

section 1 "Install packages"
sudo apt-get update && sudo apt-get install -y git-lfs bc

section 2 "Paths (from .env)"
if [ -z "\$COSMOS_REPO_DIR" ] || [ -z "\$COSMOS_HF_HOME" ] || [ -z "\$COSMOS_VENV_DIR" ] || [ -z "\$COSMOS_TORCH_HOME" ] || [ -z "\$COSMOS_PIP_CACHE_DIR" ] || [ -z "\$COSMOS_ROBOTIC_US_DIR" ] || [ -z "\$COSMOS_UV_CACHE_DIR" ]; then
  echo "Error: All COSMOS_* paths must be set in .env. Run 01-system-setup.sh or copy .env.template and fill in." >&2
  exit 1
fi
if [ -d "\$COSMOS_REPO_DIR" ] && [ ! -d "\$COSMOS_REPO_DIR/.git" ]; then
  COSMOS_REPO_DIR="\${COSMOS_REPO_DIR}/cosmos-transfer2.5"
fi
echo "COSMOS_REPO_DIR=\$COSMOS_REPO_DIR"
echo "COSMOS_HF_HOME=\$COSMOS_HF_HOME" "COSMOS_VENV_DIR=\$COSMOS_VENV_DIR" "COSMOS_TORCH_HOME=\$COSMOS_TORCH_HOME" "COSMOS_PIP_CACHE_DIR=\$COSMOS_PIP_CACHE_DIR"

section 3 "Create directories"
if [ "\$COSMOS_REPO_DIR" = "\$HOME" ] || [ "\${COSMOS_REPO_DIR#\$HOME/}" != "\$COSMOS_REPO_DIR" ]; then
  mkdir -p "\$COSMOS_HF_HOME" "\$COSMOS_VENV_DIR" "\$COSMOS_TORCH_HOME" "\$COSMOS_PIP_CACHE_DIR" "\$COSMOS_ROBOTIC_US_DIR" "\$COSMOS_UV_CACHE_DIR"
else
  sudo mkdir -p "\$COSMOS_HF_HOME" "\$COSMOS_VENV_DIR" "\$COSMOS_TORCH_HOME" "\$COSMOS_PIP_CACHE_DIR" "\$COSMOS_ROBOTIC_US_DIR" "\$COSMOS_UV_CACHE_DIR"
  sudo chown -R $RUN_USER:$RUN_GROUP "\$COSMOS_HF_HOME" "\$COSMOS_VENV_DIR" "\$COSMOS_TORCH_HOME" "\$COSMOS_PIP_CACHE_DIR" "\$COSMOS_ROBOTIC_US_DIR" "\$COSMOS_UV_CACHE_DIR"
fi
chmod -R u+rwX "\$COSMOS_HF_HOME" "\$COSMOS_TORCH_HOME" "\$COSMOS_PIP_CACHE_DIR" "\$COSMOS_UV_CACHE_DIR" 2>/dev/null || true

section 4 "GPU/CPU info"
NUM_GPUS=\$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
NUM_CPUS=\$(nproc)
echo "Detected \$NUM_GPUS GPU(s), \$NUM_CPUS CPU(s)"

section 5 "Clone repo and checkout branch"
if [ ! -d "\$COSMOS_REPO_DIR" ]; then
  echo "Cloning cosmos-transfer2.5 (branch pg/guided-gen) to \$COSMOS_REPO_DIR..."
  git clone --branch pg/guided-gen --single-branch https://github.com/guopengf/cosmos-transfer2.5.git "\$COSMOS_REPO_DIR"
else
  echo "Repository already exists at \$COSMOS_REPO_DIR"
fi
cd "\$COSMOS_REPO_DIR"
git lfs install
git lfs pull

section 6 "Build Docker image"
# Exclude .nv and other cache/restricted dirs from build context (avoids "open .nv: permission denied")
for entry in .nv .cache; do
  [ -e "\$entry" ] && { grep -qFx "\$entry" .dockerignore 2>/dev/null || echo "\$entry" >> .dockerignore; }
done
docker build --network=host --ulimit nofile=131071:131071 -f Dockerfile . -t guided-transfer2.5

echo ""
echo "Setup complete! Repository at: \$COSMOS_REPO_DIR"
echo "Run container with: \$SCRIPT_DIR/run_guided_transfer2.5_docker.sh"
echo "Attach with: \$SCRIPT_DIR/03-attach_guided_transfer2.5_docker.sh"
INNEREOF

# Run inner script as RUN_USER; report failure by section
echo "Running setup as user: $RUN_USER"
if sudo -u "$RUN_USER" bash "$INNER_SCRIPT"; then
  echo "[OK] All sections completed."
else
  echo "[FAIL] Setup failed in the section shown above." >&2
  rm -f "$INNER_SCRIPT"
  exit 1
fi
rm -f "$INNER_SCRIPT"
