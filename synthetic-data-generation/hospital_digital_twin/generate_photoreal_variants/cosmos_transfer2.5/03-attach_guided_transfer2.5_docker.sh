#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Attach to running guided-transfer2.5 container with a new bash shell

if docker ps --format '{{.Names}}' | grep -q '^guided-transfer2.5$'; then
    echo "Attaching to guided-transfer2.5 container..."
    docker exec -it guided-transfer2.5 bash
else
    echo "Error: guided-transfer2.5 container is not running"
    echo ""
    echo "Start it first with:"
    echo "  ./run_guided_transfer2.5_docker.sh"
    exit 1
fi
