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

# ADD these classes to isaaclab_arena/assets/background_library.py
#
# Required imports already present in that file:
#   from isaaclab_arena.assets.register import register_asset
#   from isaaclab_arena.utils.pose import Pose
#   from isaaclab_arena.assets.background_library import LibraryBackground
#
# Also add at the top of background_library.py:
#   from isaaclab_arena.assets.asset_paths import NUREC_ORCA_BACKGROUND

from isaaclab_arena.assets.asset_paths import NUREC_ORCA_BACKGROUND
from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


@register_asset
class NuRecOrcaBackground(LibraryBackground):
    """NuRec Orca background — Real2Sim capture of a hospital room."""

    name = "nurec_orca"
    tags = ["background"]
    usd_path = NUREC_ORCA_BACKGROUND
    initial_pose = Pose.identity()
    object_min_z = -0.2

    def __init__(self):
        super().__init__()
