# Installation

This page covers installation of the three software stacks used across this tutorial series.

> **Note for explorers:** You do **not** need to install the full Isaac ecosystem to try
> NuRec or Cosmos Transfer. Both tools run independently in their own Docker environments.
> If you want to experiment with Real2Sim reconstruction or visual augmentation before
> committing to writing Isaac applications, jump straight to
> [NuRec](#2-real2sim-reconstruction---nurec) or [Cosmos Transfer 2.5](#3-cosmos-transfer-25).

---

## 1. Isaac Ecosystem (Isaac Sim + Isaac Lab + Isaac Lab Arena)

The core simulation stack. Required for environment authoring, robot control,
teleoperation, and data collection.

**Supported versions:** Isaac Sim 5.1.0 · Isaac Lab 2.3.0

**Hardware requirements:** see [Isaac Sim Requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

### Install via Docker

**1. Clone the repository and initialise submodules:**

```bash
git clone git@github.com:isaac-sim/IsaacLab-Arena.git
cd IsaacLab-Arena
git checkout 755e8cf393165bc947fb3fb4fb07aaaa0e5dded0 # Pinned version when developing this tutorial
git submodule update --init --recursive
```

**2. Launch the Docker container:**

The docker image includes installation of IsaacSim and IsaacLab.

```bash
./docker/run_docker.sh
```

**3. (Optional) Verify the installation:**

```bash
pytest -sv -m with_cameras isaaclab_arena/tests/ --ignore=isaaclab_arena/tests/policy/
pytest -sv -m "not with_cameras" isaaclab_arena/tests/ --ignore=isaaclab_arena/tests/policy/
```

Refer to [IsaacLab-Arena document](https://isaac-sim.github.io/IsaacLab-Arena/release/0.1.1/pages/quickstart/installation.html) for more details.

With the container running, you are ready to follow [01 — Scene Creation](./01_scene_creation.md) to start adding workflows inside IsaacLab-Arena.

---

## 2. Real2Sim Reconstruction - NuRec

Real2Sim reconstruction pipeline — converts smartphone photos or videos into
simulation-ready USDZ assets. The generated USDZ file is ready to import in IsaacSim for viewing or editing.

Follow the [reconstruction-from-video installation guide](../bring_your_own_or/reconstruct_or_from_video/).

### Rendering NuRec volumes inside an IsaacLab application

If you load a NuRec Gaussian splat volume inside an IsaacLab application, you must apply
the NRE render settings right after `AppLauncher` initialises the simulator — before any
environment or scene setup:

```python
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Required to render NuRec Gaussian splat volumes correctly
import carb
carb.settings.get_settings().set("/rtx/nre/compositing/rendererHints", 3)

# Continue with the rest of your imports and env creation ...
```

Without this patch the Gaussian splat volume will not render correctly.

---

## 3. Cosmos Transfer 2.5

Visual domain augmentation for sim2real transfer.

Follow the [Cosmos Transfer 2.5 installation guide](../generate_photoreal_variants/cosmos_transfer2.5/README.md).
