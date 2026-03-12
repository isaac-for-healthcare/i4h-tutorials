# Generate Synthetic Medical Images (CT/MR)

This guide covers generating synthetic CT and MR imaging data and segmentation masks using **MAISI** (and related tools) for use in the Patient Digital Twin pipeline and downstream USD conversion.

## Learning Objectives

By the end of this guide, you will be able to:

- Use MAISI CT to generate synthetic CT data
- Understand the value of synthetic data for building autonomous healthcare robots and medical imaging research

---

## Benefits of Synthetic Data Generation (SDG)

| Benefit | Medical Imaging | Robotics Technologies |
| --------- | ---------------- | --------------------- |
| **🔍 Addressing Data Scarcity** | Medical imaging data, especially for rare diseases or edge cases, is often scarce or expensive to collect. Synthetic data fills these gaps by generating realistic images, improving data diversity and volume for training AI models. | Robotics applications often require vast, diverse datasets for training machine learning models in tasks like object recognition or motion planning. Synthetic data provides scalable, customizable data for these needs. |
| **🔒 Enhancing Privacy** | Synthetic data is not linked to real individuals and thus avoids patient privacy concerns and regulatory hurdles. This enables wider data sharing and collaboration without violating laws like HIPAA or GDPR. | In scenarios where sensor or camera data is sensitive, synthetic data can be shared more freely for cross-team or cross-institution advancement. |
| **💰 Cost and Efficiency** | Creating and annotating real medical images is time-consuming and costly. Synthetic data can be rapidly generated at lower cost, expediting the development and validation of AI tools. | Generating and annotating real-world robotics data can be prohibitively expensive. Synthetic data circumvents this by allowing efficient creation and labeling of diverse datasets. |
| **⚖️ Reducing Bias** | By generating data for underrepresented populations or rare conditions, synthetic datasets can help reduce bias in AI models, leading to fairer, more generalizable healthcare solutions. | Exposure to a broad spectrum of synthetic scenarios enhances robots' adaptability to new or unseen real-world conditions. This is crucial for applications like assistive robotics or autonomous navigation. |
| **⚡ Accelerating Innovation** | Synthetic data is used to train, validate, and benchmark AI models, speeds up clinical trials simulation, and supports medical education by providing diverse case material. | Robotic systems can be tested and trained in photorealistic or highly variable virtual environments, including edge cases that are rare or hazardous in the physical world, increasing safety and robustness. |

### Simulation Benefits

Simulation environments reduce development time and cost by enabling rapid prototyping and testing of algorithms and designs entirely in a virtual setting—eliminating the need to build and modify early physical prototypes. This approach allows software to be developed and iterated quickly, accelerates the engineering timeline, and lowers expenses related to hardware and materials.

**Software-in-the-loop (SIL)** testing lets developers validate control algorithms in a fully simulated environment, allowing fast, low-risk iterations. **Hardware-in-the-loop (HIL)** testing connects real hardware to simulated scenarios, detecting hardware-specific issues and increasing system reliability before full deployment—all while reducing the need for costly prototype builds.

---

## MAISI CT: Foundational CT Volume Generation Model

Patient anatomy examples were generated using the **MAISI foundational CT volume generation model**, which leverages generative AI to create high-quality, diverse synthetic CT data for medical imaging research and development. MAISI CT helps address data scarcity and privacy challenges in healthcare AI by providing realistic, customizable anatomical datasets.

### Resources

- [📖 Overview Blog: Addressing Medical Imaging Limitations with Synthetic Data Generation](https://developer.nvidia.com/blog/addressing-medical-imaging-limitations-with-synthetic-data-generation/)
- [📄 MAISI CT Paper (arXiv)](https://arxiv.org/html/2409.11169v1)
- [🔧 MAISI Nvidia Inference Microservice (NIM)](https://build.nvidia.com/nvidia/maisi)
- [📦 Project-MONAI/models/maisi_ct_generative (Model Zoo)](https://github.com/Project-MONAI/model-zoo/tree/dev/models/maisi_ct_generative)

---

## Run MAISI CT Pipeline Locally with MONAI Model Zoo

### 1️⃣ Clone the repo and install maisi_ct_generative

Follow the steps in the [official repository](https://github.com/Project-MONAI/model-zoo/tree/dev/models/maisi_ct_generative) to clone and install the model. The following modifications were tested on git hash: `05067dce4db8fcb87dc31e7fa510c494959230ea`

```bash
pip install "monai[fire]"
python -m monai.bundle download "maisi_ct_generative" --bundle_dir "bundles/"
```

> **💡 Tip:** The standard model requires a selection of anatomical features, though skin is not one of them. For our purposes, we can simply uncomment the filter function in `bundles/maisi_ct_generative/scripts/sample.py`. This will save all labels used during the data generation.

```bash
# synthetic_labels = filter_mask_with_organs(synthetic_labels, self.anatomy_list)
```

### 2️⃣ Adjust the config to have an empty anatomy_list

- Copy the inference script and modify it with the below instructions
- Edit the configuration file (e.g., `configs/inference_all.json`) and set `anatomy_list` to an empty list (`[]`)
- You may need to adjust additional parameters in the config to fit the model on your GPU. The file in `utils/config/inference_all.json` was used to generate the sample CTs for this course
- This ensures that all labels will be returned in the output

### 3️⃣ Run MAISI from the MONAI Model Zoo

```bash
python -m monai.bundle run --config_file configs/inference_all.json
```

### 4️⃣ Visualize generated CT data

Install Slicer SDK or another application to view the CT data and labelmap.

---

## Next steps

Generated CT/MR data and segmentation masks can be converted to USD for use in the Patient Digital Twin pipeline:

- **[Convert CT to USD](../convert_medical_images_to_USD/convert_CT_to_USD/README.md)** — NRRD/NIfTI to mesh to USD conversion for Isaac Sim.
- **[Convert CT/MR to USD with MONAI](../convert_medical_images_to_USD/convert_CT_MR_to_USD_with_MONAI/README.md)** — Bring your own patient (CT/MRI or synthetic) and convert to USD.
