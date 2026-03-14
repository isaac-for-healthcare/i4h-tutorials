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

# ADD these classes to isaaclab_arena/assets/object_library.py
#
# Required imports already present in that file:
#   from isaaclab_arena.assets.register import register_asset
#   from isaaclab_arena.assets.object_library import LibraryObject
#
# Also add at the top of object_library.py:
#   from isaaclab_arena.assets.asset_paths import ABD_PHANTOM, TABLE_WITH_COVER

from isaaclab_arena.assets.asset_paths import ABD_PHANTOM, TABLE_WITH_COVER
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


@register_asset
class ABDPhantom(LibraryObject):
    """Abdominal phantom model used in robotic ultrasound tasks."""

    name = "abd_phantom"
    tags = ["object"]
    usd_path = ABD_PHANTOM

    # Task metadata: offsets from phantom root origin in phantom local frame (x, y, z).
    # Used by UltrasoundApproachPhantomTask for success checking — no hard-coded coords needed.
    contact_point_offset: tuple[float, float, float] = (0.0, 0.0809, 0.10)
    scan_end_point_offset: tuple[float, float, float] = (-0.075, -0.062, 0.09)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class TableWithCover(LibraryObject):
    """Table with a cover."""

    name = "table_with_cover"
    tags = ["object"]
    usd_path = TABLE_WITH_COVER

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)
