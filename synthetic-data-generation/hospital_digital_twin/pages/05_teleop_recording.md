# 05 — Teleoperation and Recording

With the environment assembled — scene, embodiment, and task — it is time to put it in
motion. Before generating large synthetic datasets, you need a small set of human
demonstrations: episodes where you manually guide the robot to complete the task, recorded
as actions and observations that can later be replayed and augmented.

IsaacLab Arena ships utility scripts for both steps —
[`teleop.py`](https://github.com/isaac-sim/IsaacLab-Arena/blob/755e8cf393165bc947fb3fb4fb07aaaa0e5dded0/isaaclab_arena/scripts/teleop.py)
for free-running interaction and
[`record_demos.py`](https://github.com/isaac-sim/IsaacLab-Arena/blob/755e8cf393165bc947fb3fb4fb07aaaa0e5dded0/isaaclab_arena/scripts/record_demos.py)
for capturing demonstrations — covering common devices including keyboard, SpaceMouse,
gamepad, and XR hand-tracking. If you have your own teleoperation hardware or pipeline,
the two concepts below are what you need to wire in.

---

## Teleoperation

Teleoperation is the live loop: at every simulation step your device produces a command,
that command is passed to `env.step()`, and the result is rendered. The core loop is simple:

```python
while simulation_app.is_running():
    action = teleop_interface.advance()   # read from device
    env.step(action.repeat(env.num_envs, 1))
```

For the Franka ultrasound workflow the action is a **6-DOF delta pose** — the same format
the IK controller expects (see [02 — Embodiment](./02_embodiment.md)). The built-in script
detects automatically whether the action space includes a gripper
([L107–L110](https://github.com/isaac-sim/IsaacLab-Arena/blob/755e8cf393165bc947fb3fb4fb07aaaa0e5dded0/isaaclab_arena/scripts/teleop.py#L107-L110))
and configures the device accordingly — for the ultrasound workflow, no gripper key is
active.

The script registers an **R** callback to reset the episode
([L164–L167](https://github.com/isaac-sim/IsaacLab-Arena/blob/755e8cf393165bc947fb3fb4fb07aaaa0e5dded0/isaaclab_arena/scripts/teleop.py#L164-L167)),
which is useful for quickly discarding a bad attempt and getting a fresh object placement.

To launch the built-in teleop loop:

```bash
python isaaclab_arena/scripts/teleop.py ultrasound \
    --teleop_device keyboard \
    --embodiment franka_ultrasound
```

Supported values for `--teleop_device`: `keyboard`, `spacemouse`, `gamepad`, `xr`. See the
[IsaacLab XR documentation](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html) and [Bring Your Own XR](../bring_your_own_xr/README.md)
for XR device setup.

> **Action shape mismatch error**
>
> If you reuse IsaacLab's stock teleoperation script with the Franka ultrasound embodiment,
> you may hit an action shape mismatch. The Franka robot in this workflow has no gripper,
> but `Se3KeyboardCfg` adds a gripper dimension by default. Fix it by passing
> `gripper_term=False`:
>
> ```python
> Se3KeyboardCfg(
>     pos_sensitivity=self.pos_sensitivity,
>     rot_sensitivity=self.rot_sensitivity,
>     sim_device=self.sim_device,
>     gripper_term=has_gripper,  # False for franka_ultrasound
> )
> ```
>
> **Keyboard layout (Se3Keyboard defaults)**
>
> | Keys   | Motion                 |
> | ------ | ---------------------- |
> | W / S  | +/− X (forward / back) |
> | A / D  | +/− Y (left / right)   |
> | Q / E  | +/− Z (up / down)      |
> | Z / X  | roll                   |
> | T / G  | pitch                  |
> | C / V  | yaw                    |

---

## Recording Demonstrations

Recording wraps the same `env.step()` loop with a **RecorderManager** — an IsaacLab
component that snapshots data at each step, tracks success, and flushes accepted episodes
to HDF5. Understanding how to configure it is what lets you record the right data for your
specific workflow.

### How the recorder works

At each step, the recorder manager snapshots the current actions, observations, and
simulation state into a buffer. When an episode ends successfully the buffer is flushed to
an HDF5 file; failed attempts are discarded without writing anything. The result is one
group per successful episode (`demo_0`, `demo_1`, …) containing:

- `actions` — the raw action tensor at every step
- `initial_state` — the full simulation state at episode start, used later to seed exact
  replay or MimicGen rollouts
- `obs/` — all observation groups from the embodiment and task (proprioception, camera
  frames if `--enable_cameras` is set, etc.)

### Attaching a recorder to the environment

The recorder is configured through `env_cfg.recorders`. IsaacLab provides
`ActionStateRecorderManagerCfg` as a ready-to-use starting point that includes default
recorder terms:

| Default term | What it records |
| --- | --- |
| `InitialStateRecorder` | Full scene state at episode start (joints, object poses, …) |
| `PreStepActionsRecorder` | Raw action vector sent at each step |
| `PreStepFlatPolicyObservationsRecorder` | Concatenated `policy` observation group |
| `PostStepStatesRecorder` | Full scene state after each step |
| `PostStepProcessedActionsRecorder` | Actions after processing by each action term |

Attaching it to your environment is three lines
([L225–L228](https://github.com/isaac-sim/IsaacLab-Arena/blob/755e8cf393165bc947fb3fb4fb07aaaa0e5dded0/isaaclab_arena/scripts/record_demos.py#L225-L228)):

```python
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

env_cfg.recorders = ActionStateRecorderManagerCfg()
env_cfg.recorders.dataset_export_dir_path = "./datasets"
env_cfg.recorders.dataset_filename = "ultrasound_demos"
env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
```

`EXPORT_SUCCEEDED_ONLY` means only episodes explicitly marked successful are written to
disk — failed attempts are silently dropped on reset.

### Recording custom data

The default recorder only captures `obs_buf["policy"]` — the flat proprioceptive group.
Anything else you need (camera frames, object poses, contact forces) must be added as a
custom **RecorderTerm**. Each term is a small class with callbacks that fire at specific
points in the step/reset lifecycle (`record_pre_step`, `record_post_step`,
`record_pre_reset`, `record_post_reset`). A callback returns a `(key, value)` pair; the
key defines where the data lands in the HDF5 hierarchy.

#### Camera observations

Camera frames live in `obs_buf["camera_obs"]` and are not touched by the default recorder
even when `--enable_cameras` is set. To capture them:

```python
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

class CameraObsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "obs/camera_obs", self._env.obs_buf["camera_obs"]

@configclass
class CameraObsRecorderCfg(RecorderTermCfg):
    class_type = CameraObsRecorder

@configclass
class MyRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_camera_obs = CameraObsRecorderCfg()
```

Assign your extended manager cfg instead of the default when configuring the environment:

```python
env_cfg.recorders = MyRecorderManagerCfg()
env_cfg.recorders.dataset_export_dir_path = "./datasets"
env_cfg.recorders.dataset_filename = "ultrasound_demos"
env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
```

### How success is detected

The recorder does not terminate on the task's `success` termination term directly —
instead it extracts the success function and calls it manually at every step
([L203–L207](https://github.com/isaac-sim/IsaacLab-Arena/blob/755e8cf393165bc947fb3fb4fb07aaaa0e5dded0/isaaclab_arena/scripts/record_demos.py#L203-L207)).
This keeps the episode alive so the robot can hold its final pose, and gives you control
over a debounce window.

The debounce is `--num_success_steps`: the success condition must hold for that many
*consecutive* steps before the episode is accepted
([L344–L354](https://github.com/isaac-sim/IsaacLab-Arena/blob/755e8cf393165bc947fb3fb4fb07aaaa0e5dded0/isaaclab_arena/scripts/record_demos.py#L344-L354)).
For contact tasks like ultrasound scanning, where the probe tip may briefly satisfy the
distance threshold before settling, 5–10 steps filters out false triggers. Once the
debounce fires the episode is marked successful, exported, and the environment resets
automatically. Press **R** to abandon a bad attempt and reset manually.

---

**Next:** [06 — Trajectory Multiplication with MimicGen](./06_mimicgen.md)
