# 01 — Scene Creation

Building a hospital automation workflow starts with defining the physical world the robot
will operate in. In IsaacLab Arena, the complete simulation environment — the digital twin
— is composed of three pieces: a **Scene**, a robot **Embodiment**, and a **Task**. This
section covers the first piece.

The **Scene** is the physical world itself: a background (room, floor, furniture) and the
objects the robot interacts with. Objects have physics enabled and can be randomised on each
episode reset. The Scene does not include the robot — that comes in the next section.

In the Franka ultrasound reference workflow, the scene is a hospital room with a table and
an abdominal phantom on top of it
([`examples/ultrasound_environment.py`](../ref_workflows/franka_ultrasound/examples/ultrasound_environment.py)):

```python
background = self.asset_registry.get_asset_by_name("nurec_orca")()
phantom    = self.asset_registry.get_asset_by_name("abd_phantom")()
table      = self.asset_registry.get_asset_by_name("table_with_cover")()

phantom.set_initial_pose(Pose(position_xyz=(0, 0, 0.95),     rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0),    rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

scene = Scene(assets=[background, phantom, table])
```

The rest of this page explains how to define each of these assets for your own workflow.

---

## Scene Assets

Both the background and objects in a Scene are **USD assets** — `.usd` or `.usda` files
loaded into the simulator. Asset authoring is done in **Isaac Sim**, which provides a 3D
composer for placing meshes, configuring physics properties, and exporting USD files. Once
authored, assets are registered in IsaacLab Arena and used for data collection and training.

There are three ways to obtain USD assets:

**1. Real2Sim Reconstruction (NuRec)**
Scan a real environment with NuRec and it will produce a USD for you. This is the fastest
path to a photorealistic, real-world-matched background. The `nurec_orca` background in this
workflow was captured this way from a real hospital room.

> **Note:** NuRec currently produces visual geometry only. For physics to work correctly
> (collision, objects resting on surfaces), simple proxy collision meshes — such as a ground
> plane or table surface — need to be added on top of the reconstructed USD.

→ [NuRec documentation](../bring_your_own_or/reconstruct_or_from_video/)

**2. Isaac Sim Authoring**
Manually compose your scene in Isaac Sim: place meshes, add physics, configure lighting,
then export as a `.usd` file. This gives full control over both visuals and physics.

→ [Isaac Sim Tutorial](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
→ [Bring Your Own OR](../bring_your_own_or/setup_custom_operating_room/)

**3. Sim-Ready Assets from i4h-asset-catalog**
The Isaac for Healthcare team maintains a catalog of pre-built, sim-ready healthcare assets
including hospital rooms, OR environments, anatomical models, and healthcare robots.

→ [i4h-asset-catalog](https://github.com/isaac-for-healthcare/i4h-asset-catalog/tree/v0.3.0)

---

Once you have a USD file, the next step is to register it in IsaacLab Arena so that it can
be looked up by name when assembling the scene. The framework distinguishes between two roles
a USD asset can play — **background** or **object** — each with its own base class.

---

## Background

A background is the persistent, non-randomised layer of the scene. It defines the room
geometry, lighting, ground plane, and fixed furniture — everything that stays the same
across all episodes. It still participates in physics: gravity and collisions are active,
so objects placed on top of it rest and interact correctly.

Registering a background requires two steps: declaring the URL in `asset_paths.py` and
defining a class in `background_library.py`.

**Step 1 — add the URL constant to `isaaclab_arena/assets/asset_paths.py`**
([reference](../ref_workflows/franka_ultrasound/assets/asset_paths.py)):

```python
MY_BACKGROUND = "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/.../my_room.usd"
```

**Step 2 — register the class in `isaaclab_arena/assets/background_library.py`**
([reference](../ref_workflows/franka_ultrasound/assets/background_library.py)):

```python
from isaaclab_arena.assets.asset_paths import MY_BACKGROUND

@register_asset
class MyBackground(LibraryBackground):
    name = "my_background"
    tags = ["background"]
    usd_path = MY_BACKGROUND
    initial_pose = Pose.identity()
    object_min_z = -0.2          # lowest z at which objects are allowed to spawn
```

The `@register_asset` decorator makes this background available anywhere via
`asset_registry.get_asset_by_name("my_background")`.

---

## Objects

Objects are the interactive elements of the scene — what the robot physically manipulates
or operates on. Unlike the background, objects have full physics enabled and can be
randomised in position and orientation on each episode reset.

Objects can carry **task metadata** as class attributes — offsets, thresholds, or any
value the task needs to reference consistently. The phantom carries the probe contact and
scan endpoint offsets, which the task uses for its success check without needing to
hard-code coordinates.

**Step 1 — add the URL constant to `isaaclab_arena/assets/asset_paths.py`** (same as background).

**Step 2 — register the class in `isaaclab_arena/assets/object_library.py`**
([reference](../ref_workflows/franka_ultrasound/assets/object_library.py)):

```python
from isaaclab_arena.assets.asset_paths import ABD_PHANTOM

@register_asset
class ABDPhantom(LibraryObject):
    name = "abd_phantom"
    tags = ["object"]
    usd_path = ABD_PHANTOM

    # task metadata — used by UltrasoundApproachPhantomTask
    contact_point_offset: tuple[float, float, float] = (0.0, 0.0809, 0.10)
    scan_end_point_offset: tuple[float, float, float] = (-0.075, -0.062, 0.09)
```

---

## Assembling the Scene

With backgrounds and objects registered, the scene is assembled in the environment file
([reference](../ref_workflows/franka_ultrasound/examples/ultrasound_environment.py)):

```python
background = self.asset_registry.get_asset_by_name("my_background")()
my_object  = self.asset_registry.get_asset_by_name("my_object")()

my_object.set_initial_pose(Pose(position_xyz=(...), rotation_wxyz=(...)))

scene = Scene(assets=[background, my_object])
```

The `Scene` object is then passed to `IsaacLabArenaEnvironment` alongside the embodiment
and task. The environment builder takes care of loading all assets into the simulation and
wiring up their physics handles.

---

**Next:** [02 — Embodiment Configuration](./02_embodiment.md)
