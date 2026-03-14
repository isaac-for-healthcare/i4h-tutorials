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

# ADD these two lines to isaaclab_arena/examples/example_environments/cli.py
#
# 1. Add the import near the other environment imports:
from isaaclab_arena.examples.example_environments.ultrasound_environment import UltrasoundEnvironment

# 2. Add the entry to the ExampleEnvironments dict:
ExampleEnvironments = {
    # ... existing entries ...
    UltrasoundEnvironment.name: UltrasoundEnvironment,
}
