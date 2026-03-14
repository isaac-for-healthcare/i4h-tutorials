# 02 — Embodiment Configuration

With the scene defined, the next piece is the **Embodiment**. In IsaacLab Arena, an
embodiment bundles everything robot-specific into one place: the robot's physical
description, how it is controlled, what it observes, and how it resets between episodes.

This separation means the same scene can be reused with a different robot by simply swapping
the embodiment, and vice versa.

In the Franka ultrasound workflow, the embodiment is a Franka Panda arm fitted with a
RealSense D405 camera and an ultrasound probe (gripper removed). The full implementation is in
[`franka_ultrasound.py`](../ref_workflows/franka_ultrasound/embodiments/franka_ultrasound/franka_ultrasound.py).

---

## Assembling and Rigging the Robot

Refer to [bring your own robot](../../robot_digital_twin/bring_your_own_robot/) to assemble and rig your robot in IsaacSim.

---

## Defining the Robot

The first thing to define is the physical robot: its joints, actuators, and initial pose.
In IsaacLab, this is expressed as an `ArticulationCfg` — it loads the robot USD and sets
up physics properties
([L53–L97](../ref_workflows/franka_ultrasound/embodiments/franka_ultrasound/franka_ultrasound.py#L53-L97)):

```python
from isaaclab_arena.assets.asset_paths import FRANKA_ULTRASOUND

FRANKA_PANDA_REALSENSE_ULTRASOUND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FRANKA_ULTRASOUND,              # probe + camera assembly USD
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,                # arm holds position without fighting gravity
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={                              # pre-configured ready-to-scan posture
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            ...
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(   # high-PD gains for precise tracking
            joint_names_expr=["panda_joint[1-4]"],
            stiffness=400.0, damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            stiffness=400.0, damping=80.0,
        ),
    },
)
```

A few choices worth noting for this workflow:

- **Gravity disabled** — the arm holds position without fighting gravity, which simplifies
  control during a slow contact scan.
- **No gripper joints** — the probe is a fixed attachment; there are no finger joints to
  actuate.
- **High-PD actuators** — stiffness 400, damping 80 gives the precise position tracking
  needed for surface contact tasks.

We also add an **end-effector frame** to track the probe tip (TCP) at runtime
([L135–L163](../ref_workflows/franka_ultrasound/embodiments/franka_ultrasound/franka_ultrasound.py#L135-L163)):

```python
ee_frame: FrameTransformerCfg = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/TCP",  # probe tip prim on the robot USD
            name="end_effector",
        ),
    ],
)
```

This `FrameTransformer` continuously publishes the TCP pose in world coordinates. Both the
IK controller and the task success checks reference `ee_frame` by name — so it must point
at the correct prim in the robot USD.

Both the `robot` articulation and the `ee_frame` transformer are collected into
[`FrankaUltrasoundSceneCfg`](../ref_workflows/franka_ultrasound/embodiments/franka_ultrasound/franka_ultrasound.py#L135-L163),
which is what the environment builder merges into the simulation scene.

---

## Defining Action Space

The **action space** determines how the robot is controlled. In the ultrasound workflow,
the robot is controlled via end-effector Inverse Kinematics
([L193–L208](../ref_workflows/franka_ultrasound/embodiments/franka_ultrasound/franka_ultrasound.py#L193-L208)):

```python
@configclass
class FrankaUltrasoundActionsCfg:

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="TCP",                      # control the probe tip frame
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,           # delta pose, not absolute target
            ik_method="dls",
        ),
    )
```

Each call to `env.step(action)` — whether driven by a teleop device, a replay script, or a
trained policy — passes a **6-DOF delta pose** (Δx, Δy, Δz, Δroll, Δpitch, Δyaw) relative
to the current probe tip position.

EE-space control with differential IK is a natural fit for contact tasks like ultrasound
scanning, where you want to directly command probe tip position rather than reason about
individual joints. For workflows where joint-space control makes more sense — such as
locomotion or high-speed manipulation — IsaacLab provides alternatives including
`JointPositionActionCfg`, `JointVelocityActionCfg`, and `OperationalSpaceControllerActionCfg`.
See the [IsaacLab action manager documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html)
for the full list.

---

## Defining Observations

The observation config defines what information is available at each step. Observations are
organised into groups; the embodiment defines the `policy` group containing proprioceptive
state
([L210–L230](../ref_workflows/franka_ultrasound/embodiments/franka_ultrasound/franka_ultrasound.py#L210-L230)):

```python
@configclass
class FrankaUltrasoundObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        actions   = ObsTerm(func=mdp_isaac_lab.last_action)   # what was sent last step
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel) # 7 joint positions
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel) # 7 joint velocities
        eef_pos   = ObsTerm(func=ee_frame_pos)                # TCP position (x, y, z)
        eef_quat  = ObsTerm(func=ee_frame_quat)               # TCP orientation
```

**Camera observations** are handled separately via
[`FrankaUltrasoundCameraCfg`](../ref_workflows/franka_ultrasound/embodiments/franka_ultrasound/franka_ultrasound.py#L164-L191),
which is only activated when `enable_cameras=True` is passed at instantiation. It spawns a
room-level RGB camera (`room_cam`, 640×480) at a configurable pose overlooking the workspace.
When enabled, the `EmbodimentBase` automatically merges the camera config into the scene and
adds a corresponding `camera_obs` observation group alongside `policy`.

```python
embodiment = self.asset_registry.get_asset_by_name("franka_ultrasound")(
    enable_cameras=True   # activates room_cam + camera_obs group
)
```

> **Note:** The task injects a further observation group (`subtask_terms`) on top of these,
> carrying subtask completion signals used by MimicGen. The embodiment has no knowledge of
> those signals — they are added by the task (covered in `03_task.md`).

---

## How it Resets

At the start of each episode, the robot arm is reset to its default joint pose with a small
random offset. This is configured in the event config
([L258–L275](../ref_workflows/franka_ultrasound/embodiments/franka_ultrasound/franka_ultrasound.py#L258-L275)):

```python
@configclass
class FrankaUltrasoundEventCfg:

    reset_joint_position = EventTerm(
        func=_reset_panda_joints_by_fraction_of_limits,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
            "fraction": 0.01,   # 1% of joint range — small but enough for diversity
        },
    )
```

The 1% joint randomisation ensures that demos are collected from slightly different starting
configurations, which helps generalise across varied robot starting poses.

---

## Wiring it Together

Once all the configs are defined, the embodiment class wires them together and registers
itself with `@register_asset`
([L104–L127](../ref_workflows/franka_ultrasound/embodiments/franka_ultrasound/franka_ultrasound.py#L104-L127)):

```python
@register_asset
class FrankaUltrasoundEmbodiment(EmbodimentBase):

    name = "franka_ultrasound"

    def __init__(self, enable_cameras=False, ...):
        self.scene_config       = FrankaUltrasoundSceneCfg()
        self.action_config      = FrankaUltrasoundActionsCfg()
        self.observation_config = FrankaUltrasoundObservationsCfg()
        self.event_config       = FrankaUltrasoundEventCfg()
        self.mimic_env          = FrankaUltrasoundMimicEnv
```

After this, the embodiment is available by name anywhere in the codebase:

```python
embodiment = self.asset_registry.get_asset_by_name("franka_ultrasound")()
```

Finally, expose the new embodiment by adding one line to
`isaaclab_arena/embodiments/__init__.py`
([reference](../ref_workflows/franka_ultrasound/embodiments/__init__.py)):

```python
from .franka_ultrasound.franka_ultrasound import *   # ← add this line
```

---

**Next:** [03 — Task Definition](./03_task.md)
