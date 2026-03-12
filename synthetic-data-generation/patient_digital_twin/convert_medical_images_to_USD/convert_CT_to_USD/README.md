# 🏥 Medical Data Conversion

## 🎯 Learning Objectives

By the end of this chapter, you will be able to:

- Convert CT data to USD format
- Understand the value of synthetic data for building autonomous healthcare robots

> **Generate synthetic CT/MR data:** For generating synthetic CT (and related) imaging data and segmentation masks with MAISI, see [Generate Synthetic Medical Images (CT/MR)](../../generate_synthetic_medical_images/README.md).

---

## 🤔 Why Convert a CT Dataset from NIfTI or DICOM to USD?

Medical imaging data, such as CT scans, are typically stored in formats like **NIfTI (.nii, .nii.gz)** or **DICOM (.dcm)**. These formats are well-suited for clinical and research workflows, as they efficiently store volumetric (3D) data and associated metadata. However, they are not directly compatible with 3D simulation, visualization, or robotics platforms like NVIDIA IsaacSim.

**Universal Scene Description (USD)** is a powerful, extensible 3D file format developed by Pixar and widely adopted in the visual effects, animation, and simulation industries. USD is designed for efficient scene representation, asset interchange, and real-time rendering.

### 🎯 Key Reasons for Conversion

- **🔄 Interoperability with Simulation Platforms:**
  IsaacSim and other robotics/graphics tools natively support USD for importing, manipulating, and rendering 3D assets. Converting medical data to USD enables seamless integration into these environments.

- **🔲 Mesh Representation:**
  NIfTI and DICOM store volumetric data (voxels), but simulation and visualization platforms require surface meshes (e.g., OBJ, STL, or USD) to represent anatomical structures. The conversion process extracts and processes these meshes from the volumetric data.

- **⚡ Efficient Rendering and Manipulation:**
  USD supports hierarchical scene graphs, material definitions, and efficient rendering pipelines, making it ideal for interactive applications, simulation, and digital twin workflows.

- **📊 Rich Metadata and Structure:**
  USD allows for the inclusion of semantic labels, hierarchical organization, and physical properties, which are essential for robotics, AI training, and advanced visualization.
  > 📚 **Learn more about USD:** [NVIDIA OpenUSD Learning Path](https://www.nvidia.com/en-us/learn/learning-path/openusd/)

### 📋 Summary

Converting CT datasets from NIfTI or DICOM to USD is necessary to:

- ✅ Enable use in simulation and robotics platforms like IsaacSim
- ✅ Transform volumetric medical data into usable 3D surface meshes
- ✅ Leverage the advanced features and performance of the USD ecosystem

---

## 🔄 Conversion to USD Format

The tool implements the complete conversion workflow:

1. **NRRD → NIfTI:** Format conversion for medical data
2. **NIfTI → Mesh:** Surface extraction with smoothing and reduction
3. **Mesh → USD:** Final format suitable for IsaacSim and 3D visualization

## 🛠️ Setup NRRD to USD Converter Tool

This project includes an `environment.yml` file for easy conda environment setup. To set up the environment and install all required dependencies for the CT-to-USD conversion workflow, you have two options:

### Option 1: Use Existing i4h-workflows Conda Environment

If you already have the i4h-workflows conda environment set up, you can install the dependencies directly:

```bash
# Activate the existing i4h environment
conda activate i4h

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### Option 2: Create a New Dedicated Conda Environment

Create a dedicated environment for the CT-to-USD converter:

```bash
# Create and activate new environment
conda env create -f environment.yml
conda activate ct-to-usd-converter
```

### 1️⃣ **NRRD to NIfTI Conversion**

- Convert segmentation files from NRRD to NIfTI format

### 2️⃣ **NIfTI to Mesh Processing**

- Groups 140 labels into 17 anatomical categories
- Creates separate OBJ files for each organ
- Applies smoothing and mesh reduction
- Supports detailed anatomical structures including:
  - 🫀 **Organs:** Liver, Spleen, Pancreas, etc.
  - 🦴 **Skeletal system:** Spine, Ribs, Shoulders, Hips
  - 🩸 **Vascular system:** Veins and Arteries
  - 💪 **Muscular system:** Back muscles

### 3️⃣ **Mesh to USD Conversion**

- Converts the processed meshes to USD format
- Final USD files can be imported into IsaacSim

---

### 🚀 Usage

**Basic Command:**

```bash
# Make sure your conda environment is activated first
conda activate ct-to-usd-converter  # or conda activate i4h

# Then run the converter
python utils/converter.py /path/to/your/ct_folder
```

### 📁 Output Structure

The converter generates:

```text
output/
├── nii/           # Intermediate NIfTI files
├── obj/           # Intermediate mesh files
└── all_organs.usd # Final USD file
```

### 🏥 Supported Anatomical Structures

The tool processes 140 labels grouped into 17 categories including:

- **🫀 Organs:** Liver, Spleen, Pancreas, Heart, Gallbladder, Stomach, Kidneys
- **🍽️ Digestive:** Small bowel, Colon
- **🦴 Skeletal:** Spine, Ribs, Shoulders, Hips
- **🩸 Vascular:** Veins and Arteries
- **🫁 Respiratory:** Lungs
- **💪 Muscular:** Back muscles

---

## 🎮 Load CT-derived data into IsaacSim

### 📋 Steps

1. **🚀 Launch IsaacSim again**
2. **📂 Find the path of your I4H-assets**
3. **📁 Open the CT-derived USD file**

---

## 🎉 Summary

This chapter has covered converting medical imaging data (CT) to USD format for use in IsaacSim and other robotics platforms. For generating synthetic CT/MR data with MAISI, see [Generate Synthetic Medical Images (CT/MR)](../../generate_synthetic_medical_images/README.md).
