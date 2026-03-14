# 06 — Trajectory Multiplication with MimicGen

Collecting human demonstrations is expensive. MimicGen solves this by taking a small set of
annotated demonstrations and automatically transferring them to thousands of new object
configurations — producing a large synthetic dataset without additional human effort.

The pipeline has three steps:

1. **Configure** — implement the MimicGen interface in the embodiment and task
2. **Annotate** — replay recorded demos and mark subtask boundaries
3. **Generate** — run MimicGen to produce the synthetic dataset

---

## Step 1 — Configure MimicGen

MimicGen requires two pieces of configuration: the **embodiment** must expose how to read
and set the robot's end-effector pose, and the **task** must declare the subtask structure.

### Embodiment interface

`FrankaUltrasoundMimicEnv` extends `ManagerBasedRLMimicEnv` and answers three questions
MimicGen asks at runtime
([L276](../ref_workflows/franka_ultrasound/embodiments/franka_ultrasound/franka_ultrasound.py#L276)):

- **Where is the probe tip?** → `get_robot_eef_pose` reads `eef_pos`/`eef_quat` from the
  observation buffer and returns a 4×4 pose matrix.
- **Given a target pose, what action do I send?** → `target_eef_pose_to_action` computes
  the delta position and axis-angle rotation between current and target pose.
- **What is the gripper doing?** → `actions_to_gripper_actions` returns empty tensors —
  there is no gripper.

If you adapt this workflow to a different robot, these three methods are what you need to
reimplement. The embodiment wires in the class via:

```python
self.mimic_env = FrankaUltrasoundMimicEnv
```

### Task config

MimicGen needs to know the subtask structure: how many subtasks there are, what object each
subtask is relative to, and how to select source segments for transfer. This is defined in a
`MimicEnvCfg` subclass inside the task file.

The ultrasound task has two subtasks:

1. **reach** — move the probe to the contact point on the phantom
2. **scan** — slide from the contact point to the scan endpoint

([reference](../ref_workflows/franka_ultrasound/tasks/ultrasound_approach_phantom_task.py)):

```python
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig

@configclass
class UltrasoundApproachPhantomMimicEnvCfg(MimicEnvCfg):

    embodiment_name: str = "franka_ultrasound"
    phantom_name: str = "abd_phantom"

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "demo_src_ultrasound_approach_phantom"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.max_num_failures = 25

        subtask_configs = [
            # Subtask 1: reach — ends when "reach_1" signal fires.
            # Annotator presses S at this moment during manual annotation.
            SubTaskConfig(
                object_ref=self.phantom_name,
                subtask_term_signal="reach_1",
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.001,
                num_interpolation_steps=5,
                num_fixed_steps=1,
            ),
            # Subtask 2: scan — final subtask, no term signal needed.
            SubTaskConfig(
                object_ref=self.phantom_name,
                subtask_term_signal=None,
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.001,
                num_interpolation_steps=5,
                num_fixed_steps=0,
            ),
        ]
        self.subtask_configs["robot"] = subtask_configs
```

The task wires this config in via `get_mimic_env_cfg()`:

```python
class UltrasoundApproachPhantomTask(TaskBase):
    def get_mimic_env_cfg(self, embodiment_name: str):
        return UltrasoundApproachPhantomMimicEnvCfg(
            embodiment_name=embodiment_name,
            phantom_name=self.phantom.name,
        )
```

Each `SubTaskConfig` defines:

- `object_ref` — the object this subtask is relative to (used for pose transfer)
- `subtask_term_signal` — which observation signal marks the end of this subtask (`None` for the final subtask)
- `selection_strategy` — how to pick the source demo segment for transfer (`nearest_neighbor_object` picks the demo whose object pose is closest to the target)
- `action_noise` — small noise added to generated actions for diversity

---

## Step 2 — Annotate Demonstrations

Before MimicGen can generate data, each recorded demo must be annotated with **subtask
boundaries** — timestamps indicating when each subtask completes. This is done by replaying
demos and pressing `S` at the moment each subtask ends.

```bash
python isaaclab_arena/scripts/annotate_demos.py \
    --input_file  ./datasets/ultrasound_demos.hdf5 \
    --output_file ./datasets/ultrasound_demos_annotated.hdf5 \
    ultrasound
```

**Keyboard controls during annotation:**

| Key   | Action                                                       |
| ----- | ------------------------------------------------------------ |
| `N`   | Play                                                         |
| `B`   | Pause                                                        |
| `S`   | Mark subtask boundary (press once per subtask transition)    |
| `Q`   | Skip this episode                                            |

For the ultrasound task, press `S` once — when the probe first makes contact with the
phantom surface (end of subtask 1 / reach). The scan subtask runs until end of episode.

> Use `--auto` to skip manual annotation when the task's success signals already fire
> reliably in replay. The script will detect boundaries automatically.

```bash
python isaaclab_arena/scripts/annotate_demos.py \
    --auto \
    --input_file  ./datasets/ultrasound_demos.hdf5 \
    --output_file ./datasets/ultrasound_demos_annotated.hdf5 \
    ultrasound
```

---

## Step 3 — Generate the Dataset

With annotated demos in hand, run MimicGen to transfer subtask segments to new object
configurations:

```bash
python isaaclab_arena/scripts/generate_dataset.py \
    --generation_num_trials 10 \
    --num_envs 10 \
    --input_file  ./datasets/ultrasound_demos_annotated.hdf5 \
    --output_file ./datasets/ultrasound_generated.hdf5 \
    --mimic \
    --enable_cameras \
    ultrasound
```

Key arguments:

| Argument                     | Description                                      |
| ---------------------------- | ------------------------------------------------ |
| `--generation_num_trials`    | Total number of demos to generate                |
| `--num_envs`                 | Parallel environments to run simultaneously      |
| `--input_file`               | Annotated source demos from Step 2               |
| `--output_file`              | Where to write the generated dataset             |
| `--mimic`                    | Enable MimicGen generation mode                  |
| `--enable_cameras`           | Record camera observations alongside state       |

The output HDF5 file has the same structure as recorded demos and can be used directly for
robot learning.

---
