# 04 — Launching the Environment

With the scene, embodiment, and task defined, it is time to assemble them into a single
simulation environment and launch it. This is a critical checkpoint — before recording
any demonstrations, you want to confirm that assets load correctly, the robot spawns in
the right pose, and objects are placed as expected.

---

## Assembling the Environment

The three pieces are passed to `IsaacLabArenaEnvironment`, which holds the configuration
before handing it to the IsaacLab environment builder:

```python
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()

# Instantiate assets
background  = asset_registry.get_asset_by_name("nurec_orca")()
phantom     = asset_registry.get_asset_by_name("abd_phantom")()
table       = asset_registry.get_asset_by_name("table_with_cover")()
embodiment  = asset_registry.get_asset_by_name("franka_ultrasound")()

# Set initial poses
phantom.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.95),       rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
embodiment.set_initial_pose(Pose(position_xyz=(0.04, -0.50, 0.83), rotation_wxyz=(0.707, 0.0, 0.0, 0.707)))

# Assemble scene (robot is NOT part of the scene)
scene = Scene(assets=[background, phantom, table])

# Wire together
env_config = IsaacLabArenaEnvironment(
    name="my_env",
    embodiment=embodiment,
    scene=scene,
    task=UltrasoundApproachPhantomTask(phantom=phantom),
)
```

---

## Compiling and Running

`ArenaEnvBuilder` takes the `IsaacLabArenaEnvironment` config and produces a standard
IsaacLab `ManagerBasedRLEnvCfg`, which is then registered and instantiated as a Gymnasium
environment
([`compile_env_notebook.py`](https://github.com/isaac-sim/IsaacLab-Arena/blob/755e8cf393165bc947fb3fb4fb07aaaa0e5dded0/isaaclab_arena/examples/compile_env_notebook.py)):

```python
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

args_cli = get_isaaclab_arena_cli_parser().parse_args([])
env_builder = ArenaEnvBuilder(env_config, args_cli)
env = env_builder.make_registered()
env.reset()
```

From this point `env` is a standard Gymnasium environment. You can step through it with
zero actions to verify physics settle correctly:

```python
import torch, tqdm

for _ in tqdm.tqdm(range(200)):
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(actions)
```

---

## Using DummyTask During Development

If you have not written your task yet — or want to verify the scene and embodiment
independently — you can substitute `DummyTask`, which provides no-op terminations and
events:

```python
from isaaclab_arena.tasks.dummy_task import DummyTask

env_config = IsaacLabArenaEnvironment(
    name="my_env",
    embodiment=embodiment,
    scene=scene,
    task=DummyTask(),   # ← placeholder until your task is ready
)
```

This lets you iterate on scene layout and robot configuration without needing a complete
task definition first.

---

## Writing an Example Environment

The `ArenaEnvBuilder` / notebook approach is useful for quick iteration, but to use your
environment with the IsaacLab Arena scripts (`teleop.py`, `record_demos.py`, etc.) you
need to wrap it in an **ExampleEnvironment** class. This is the thin adapter that the CLI
knows how to find and instantiate.

Subclass `ExampleEnvironmentBase` and implement `get_env`, which receives the parsed CLI
args and returns a fully configured `IsaacLabArenaEnvironment`
([reference](../ref_workflows/franka_ultrasound/examples/ultrasound_environment.py)):

```python
# isaaclab_arena/examples/example_environments/my_environment.py
import argparse
from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase

class MyEnvironment(ExampleEnvironmentBase):

    name: str = "my_env"   # the identifier used on the command line

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from my_workflow.task import MyTask

        embodiment  = self.asset_registry.get_asset_by_name("franka_ultrasound")(
            enable_cameras=args_cli.enable_cameras
        )
        background  = self.asset_registry.get_asset_by_name("nurec_orca")()
        my_object   = self.asset_registry.get_asset_by_name("abd_phantom")()
        table       = self.asset_registry.get_asset_by_name("table_with_cover")()

        my_object.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.95), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        scene = Scene(assets=[background, my_object, table])

        teleop_device = (
            self.device_registry.get_device_by_name(args_cli.teleop_device)()
            if args_cli.teleop_device else None
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=MyTask(my_object=my_object),
            teleop_device=teleop_device,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--enable_cameras", action="store_true", default=False)
```

A few things to note:

- All asset and task imports happen **inside** `get_env`, not at module level. The CLI
  imports this file before Isaac Sim starts, so top-level imports of simulation-dependent
  modules will crash.
- `add_cli_args` declares any environment-specific flags. At minimum you almost always
  want `--teleop_device`.

---

## Registering in the CLI

With the environment class written, register it in
`isaaclab_arena/examples/example_environments/cli.py`
([reference](https://github.com/isaac-sim/IsaacLab-Arena/blob/755e8cf393165bc947fb3fb4fb07aaaa0e5dded0/isaaclab_arena/examples/example_environments/cli.py)):

```python
# add at the top with the other imports
from isaaclab_arena.examples.example_environments.my_environment import MyEnvironment

# add to the ExampleEnvironments dict
ExampleEnvironments = {
    ...
    MyEnvironment.name: MyEnvironment,
}
```

That is all. `MyEnvironment.name` (`"my_env"`) is now a valid positional argument for
every script in `isaaclab_arena/scripts/`.

---

## Using the Environment with Scripts

Once registered, the environment name is passed as a positional argument to any script:

```bash
# interactive teleoperation
python isaaclab_arena/scripts/teleop.py my_env --teleop_device keyboard

# record demonstrations
python isaaclab_arena/scripts/record_demos.py \
    --dataset_file ./datasets/my_demos.hdf5 \
    --num_demos 10 \
    my_env --teleop_device keyboard

# replay recorded demos
python isaaclab_arena/scripts/replay_demos.py \
    --dataset_file ./datasets/my_demos.hdf5 \
    my_env
```

The scripts build the environment by calling `MyEnvironment().get_env(args_cli)` and
passing the result through `ArenaEnvBuilder` — the same path as the notebook approach,
just automated.

---

**Next:** [05 — Teleoperation and Recording](./05_teleop_recording.md)
