# 03 — Task Definition

With the scene and embodiment in place, the final piece of the simulation environment is
the **Task** — the definition of what the robot is supposed to do, and how we know when it
has done it.

> **Note:**
> You can skip ahead to [04 — Environment](./04_environment.md) to quickly verify scene + robot using a `DummyTask`
> before writing a full task.

A task in IsaacLab Arena defines:

- **When an episode ends** — success condition and timeout
- **What changes on each reset** — randomisation of objects in the scene
- **Subtask signals** — intermediate checkpoints used by MimicGen to structure demonstrations
- **Metrics** — how performance is measured

In the Franka ultrasound workflow, the task is a two-phase scan: the robot must first bring
the probe tip to a contact point on the phantom surface (`reach`), then slide it to a scan
end point (`scan`). The full implementation is in
[`ultrasound_approach_phantom_task.py`](../ref_workflows/franka_ultrasound/tasks/ultrasound_approach_phantom_task.py).

---

## Defining Success and Termination

The first question a task must answer is: *when does an episode end?*

For the ultrasound task, the episode succeeds when the probe tip reaches the scan end point
on the phantom surface. This is checked by `probe_scan_phantom`
([L75–L86](../ref_workflows/franka_ultrasound/tasks/ultrasound_approach_phantom_task.py#L75-L86)),
which computes the distance between the `ee_frame` TCP position and a phantom-local offset
point:

```python
def probe_scan_phantom(env, ee_frame_cfg, phantom_cfg, scan_end_point_offset, threshold=0.05):
    """Returns True when the probe EEF is within threshold metres of the scan end point."""
    ...
```

This function is wired into the termination config as the `success` term
([L102–L115](../ref_workflows/franka_ultrasound/tasks/ultrasound_approach_phantom_task.py#L102-L115)).
Notice that `scan_end_point_offset` is read directly from the phantom object — the same
attribute defined back in `01_scene_creation.md`:

```python
def get_termination_cfg(self):
    success = TerminationTermCfg(
        func=probe_scan_phantom,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "phantom_cfg": SceneEntityCfg(self.phantom.name),
            "scan_end_point_offset": self.phantom.scan_end_point_offset,  # from ABDPhantom
        },
    )
    return TerminationsCfg(success=success)
```

A `time_out` term is also included automatically via `TerminationsCfg` so that episodes
that stall still eventually reset.

---

## Randomising the Scene on Reset

To make the collected data diverse, the phantom is randomly displaced on each episode
reset. This is defined in `EventsCfg`
([L165–L182](../ref_workflows/franka_ultrasound/tasks/ultrasound_approach_phantom_task.py#L165-L182)):

```python
@configclass
class EventsCfg:

    reset_phantom_pose = EventTermCfg(
        func=mdp_isaac_lab.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.025, 0.025),     # small lateral shift
                "y": (0.10, 0.12),        # in front of the robot
                "yaw": (-pi/8, pi/8),     # random rotation around vertical axis
            },
            "asset_cfg": SceneEntityCfg(phantom.name),
        },
    )
```

Each episode the phantom lands at a slightly different position and orientation. Because the
success check and subtask signals use phantom-local offsets (`contact_point_offset`,
`scan_end_point_offset`), they automatically follow the phantom wherever it resets — no
manual pose tracking needed.

---

## Subtask Signals

The ultrasound scan has two phases. MimicGen needs to know where one ends and the other
begins in each recorded demonstration. This boundary is expressed as a **subtask signal**
— an observation term that fires `True` when the robot completes the first subtask.

The signal `reach_1` is defined as an observation in `get_observation_cfg`
([L125–L138](../ref_workflows/franka_ultrasound/tasks/ultrasound_approach_phantom_task.py#L125-L138)).
It uses `probe_reach_phantom`, which mirrors the success check but tests against
`contact_point_offset` instead of `scan_end_point_offset`:

```python
def get_observation_cfg(self):
    subtask_terms = UltrasoundSubtaskTermsCfg(
        reach_1=ObsTerm(
            func=probe_reach_phantom,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "phantom_cfg": SceneEntityCfg(self.phantom.name),
                "contact_point_offset": self.phantom.contact_point_offset,
            },
        )
    )
    return UltrasoundTaskObservationsCfg(subtask_terms=subtask_terms)
```

This observation group (`subtask_terms`) is what the embodiment's `get_subtask_term_signals`
reads during MimicGen data generation. The task injects it on top of the embodiment's
`policy` observations — neither component needs to know about the other's internals.

---

## Metrics

The task reports a `SuccessRateMetric` — the fraction of episodes that terminated with
`success` rather than `time_out`
([L147–L149](../ref_workflows/franka_ultrasound/tasks/ultrasound_approach_phantom_task.py#L147-L149)).
This is logged automatically during teleoperation, replay, and data generation runs.

> The MimicGen config (`MimicEnvCfg`, `SubTaskConfig`) and the embodiment interface
> (`FrankaUltrasoundMimicEnv`) are covered in [06 — MimicGen](./06_mimicgen.md).

---

## Wiring it Together

The task is instantiated with the phantom object and passed to `IsaacLabArenaEnvironment`.
The environment assembly also needs to be registered in the CLI.

**Create `isaaclab_arena/examples/example_environments/<your_env>.py`**
([reference](../ref_workflows/franka_ultrasound/examples/ultrasound_environment.py)):

```python
class MyEnvironment(ExampleEnvironmentBase):
    name = "my_env"

    def get_env(self, args_cli):
        embodiment = self.asset_registry.get_asset_by_name("my_robot")(
            enable_cameras=args_cli.enable_cameras
        )
        background = self.asset_registry.get_asset_by_name("my_background")()
        my_object  = self.asset_registry.get_asset_by_name("my_object")()
        scene = Scene(assets=[background, my_object])
        task = MyTask(my_object=my_object)

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=self.device_registry.get_device_by_name(args_cli.teleop_device)(),
        )
```

**Add two lines to `isaaclab_arena/examples/example_environments/cli.py`**
([reference](../ref_workflows/franka_ultrasound/examples/cli_addition.py)):

```python
# at the top, with the other imports:
from isaaclab_arena.examples.example_environments.my_environment import MyEnvironment

# in the ExampleEnvironments dict:
ExampleEnvironments = {
    ...
    MyEnvironment.name: MyEnvironment,   # ← add this line
}
```

The environment builder merges the task's terminations, events, observations, and MimicGen
config with those from the scene and embodiment to produce the final `ManagerBasedRLEnvCfg`.

---

**Next:** [04 — Launching the Environment](./04_environment.md)
