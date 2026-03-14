# Isaac for Healthcare Synthetic Data Generation

This repository contains tutorials and reusable components for customizing and extending Isaac for Healthcare workflows with your own assets, data, and environments. The content is organized around **digital twin** pipelines (patient, hospital, robot) and **surgical robotic** simulation and video generation.

## [Patient Digital Twin](patient_digital_twin/)

Turn clinical data (imaging, physiological) into simulation-ready 3D assets in Universal Scene Description (USD). Covers synthetic medical image generation, CT/MR to USD conversion, material assignment, and style augmentation.

- [Generate imaging data and segmentation masks (NV-Generate)](./patient_digital_twin/generate_synthetic_medical_images/README.md)
- [Convert CT/MR to USD with MONAI](./patient_digital_twin/convert_medical_images_to_USD/convert_CT_MR_to_USD_with_MONAI/README.md)
- [Convert CT to USD with local tools](./patient_digital_twin/convert_medical_images_to_USD/convert_CT_to_USD/README.md)

## [Hospital Digital Twin](hospital_digital_twin/)

Turn real-world or authored environments into simulation-ready environment: environment creation, robot rigging, teleoperation, recording, trajectory generation, and visual style augmentation.

- [Environment creation](hospital_digital_twin/README.md#-environment-creation) — Bring your own OR, robot rigging
- [Data Collection](hospital_digital_twin/README.md#data-collection) — Teleop, Bring your own XR, recording
- [Data Augmentation](hospital_digital_twin/README.md#data-generation) — Mimicgen, cosmos-transfer

## [Robot Digital Twin](robot_digital_twin/)

Bring your own robot into the pipeline: custom URDF/CAD, rigging in Isaac Sim, and integration with hospital and patient digital twins.

- [Bring your own robot](./robot_digital_twin/bring_your_own_robot/README.md) — e.g. [Replace Franka hand with ultrasound probe](./robot_digital_twin/bring_your_own_robot/README.md)

## [Surgical Robotic Video Generator](surgical_robotic_video_generator/)

Bridge the Cosmos-H-Surgical-Predict world model with downstream policy (inverse kinematic models) and use Cosmos-H-Surgical-Transfer for data augmentation and policy generalizability. See the [SurgWorld paper](https://arxiv.org/abs/2512.23162).

## [Surgical Robotic Generative Physics Simulator](surgical_robotic_generative_physics_simulator/)

Post-train (finetune) Cosmos-H-Surgical-Simulator on a custom surgical robotics dataset for policy evaluation and synthetic data generation. Includes Slurm/multi-node setup and SutureBot as an example custom dataset.
