# Reference Workflow: Franka Ultrasound

Complete, copy-paste-ready code for the Franka ultrasound workflow.
Use this alongside the tutorial docs to build your own workflow.

## What to do with each file

| File in this folder | Action | Destination in `isaaclab_arena/` |
| --- | --- | --- |
| `assets/asset_paths.py` | **Copy** the file, update URLs | `assets/asset_paths.py` (new file) |
| `assets/background_library.py` | **Add** the class shown | `assets/background_library.py` |
| `assets/object_library.py` | **Add** the two classes shown | `assets/object_library.py` |
| `embodiments/__init__.py` | **Add** the one import line | `embodiments/__init__.py` |
| `embodiments/franka_ultrasound/` | **Copy** the whole folder | `embodiments/franka_ultrasound/` |
| `tasks/ultrasound_approach_phantom_task.py` | **Copy** the file | `tasks/` |
| `examples/ultrasound_environment.py` | **Copy** the file | `examples/example_environments/` |
| `examples/cli_addition.py` | **Add** the two lines shown | `examples/example_environments/cli.py` |

For files marked **Add**, only paste the shown code — do not replace the whole file.
For files marked **Copy**, place the file directly in the destination folder.
