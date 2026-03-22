# CACR-Net

Official repository for the paper **CACR-Net: Constraint-Aware Occlusion-Guided Dental Crown Reconstruction with Latent SDF Diffusion**.

CACR-Net is a two-stage framework for reconstructing anatomically accurate and functionally plausible dental crowns from intraoral scans. It combines an occlusion-aware point-cloud reconstruction network with latent signed distance field diffusion to generate closed crown meshes that can be integrated into CAD/CAM workflows.

## Overview

Dental crown reconstruction faces three practical challenges:

- limited methods that generalize across all tooth types
- insufficient 3D modeling fidelity from intraoral scans
- lack of explicit constraints on occlusal morphology and non-penetration

CACR-Net addresses these issues with two stages:

1. **CMDen-Net** predicts an initial crown point cloud from a single-arch intraoral scan.
2. **SDFDiff-Net** refines the reconstruction in a latent SDF space and decodes a closed triangular crown mesh.

## Method

### Stage 1: CMDen-Net

- **Tooth-wise Iterative Farthest Point Sampling (TIFPS)** builds three tooth-aware point sets (high, medium, and low resolution) by applying iterative FPS independently within each tooth region.
- **Voxelization** maps normalized coordinates to a 64³ voxel grid. Four serialization schemes (Hilbert, Z-order, raster, zigzag) order voxels into 1-D sequences.
- **Serialized-Mamba Network (SMN)** runs four Mamba blocks, each randomly selecting a serialization scheme, to model long-range geometric dependencies across the full arch. An **Order Index Prompt (OIP)** is inserted at both ends of each serialized sequence so the network can distinguish traversal schemes.
- **Dual-branch geometric enhancement** separately encodes coordinates (via pointwise MLP + max-pool) and normals (via normal-variation score MLP + max-pool), then fuses the two branches at the point level.
- **Hierarchical decoding** concatenates multi-resolution SMN features into a global feature F, then decodes three point clouds (P_L → P_M → P_H) in a pyramid fashion inspired by FPN.
- **Curvature-Guided Occlusal Constraint (CGOC)** selects the top-30% highest-curvature points (cusps, grooves, fossae) and supervises them with soft-matched positional and curvature differences.
- **SDF-based non-penetration loss** penalizes predicted crown points that fall inside the environment mesh (neighboring dentition + antagonist teeth).

### Stage 2: SDFDiff-Net

- A **variational autoencoder (VAE)** with latent dimension 512 learns a decodable latent SDF space on ground-truth crowns.
- **Conditional DDPM diffusion** (T = 800 steps) is performed in the VAE latent space, conditioned on the stage-1 predicted crown point cloud and antagonist teeth encoded by a PointNet-based encoder.
- **FiLM-modulated Mamba layers** in the denoising network receive the concatenated conditioning + timestep embedding as scale/shift factors.
- **Marching Cubes** extracts the zero-level set on a 128³ grid as a CAD/CAM-ready closed triangular mesh.

## Main Contributions

- A two-stage occlusion-guided framework for dental crown reconstruction across all tooth types.
- A Serialized-Mamba point-cloud encoder with Order Index Prompting for efficient multi-tooth geometric modeling.
- Explicit occlusal and collision constraints through CGOC and SDF-based non-penetration supervision.
- A latent conditional SDF diffusion stage for generating closed crown meshes directly usable in CAD/CAM.

## Results

### Dataset

Experiments use the public [Teeth3DS+ dataset](https://osf.io/xctdy/), which contains 1,800 intraoral scans from 900 patients with paired upper- and lower-jaw point clouds. Scans were captured by three intraoral scanners (Primescan, Dentsply Sirona; Trios3, 3Shape; iTero Element 2 Plus, Align Technology) at 10–90 µm spatial resolution and anonymized to comply with GDPR. Teeth are labeled in the FDI two-digit numbering system. The dataset is split at the patient level into training / validation / testing sets (70% / 10% / 20%) with disjoint patients. Each crown is represented by 2,048 uniformly sampled points normalized to [0, 1].

### Quantitative Results

> All table values are computed in normalized [0, 1] space and multiplied by 1000. Denormalized physical-space errors are **0.7392 mm (CD-L1)** and **0.5923 mm (CD-L2)**.

**Performance by tooth category (Teeth3DS+, FDI):**

| Category | FDI IDs | CD-L1 | CD-L2 | EMD | MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Incisors | 11–12, 21–22, 31–32, 41–42 | 12.79 | 2.17 | 29.73 | 0.0016 |
| Canines | 13, 23, 33, 43 | 13.57 | 2.45 | 30.87 | 0.0020 |
| Premolars | 14–15, 24–25, 34–35, 44–45 | 14.12 | 2.48 | 31.76 | 0.0023 |
| Molars | 16–18, 26–28, 36–38, 46–48 | 15.04 | 2.69 | 31.92 | 0.0026 |
| **Overall** | All test cases | **13.92** | **2.31** | **30.22** | **0.0018** |

Errors are lower for simpler anterior teeth and higher for morphologically complex posterior molars, as expected.

**Comparison with existing methods:**

| Method | CD-L1 | CD-L2 | EMD | MSE |
| --- | ---: | ---: | ---: | ---: |
| PF-Net | 26.64 | 7.64 | 41.36 | – |
| FSC | 20.49 | 5.51 | 38.92 | – |
| AdaPoinTr | 19.11 | 4.80 | 36.29 | – |
| PDCNet\* | 54.39 | 8.41 | 75.31 | 0.0023 |
| **CACR-Net (Ours)** | **13.92** | **2.31** | **30.22** | **0.0018** |

PF-Net, FSC, and AdaPoinTr were retrained and evaluated on Teeth3DS+ under the same preprocessing pipeline and data split. MSE (–) is not applicable to point-only baselines.

\* PDCNet values are from the published paper only; its code was not available for re-evaluation under the same protocol, and its larger distances reflect unnormalized point-cloud metrics.

### Ablation Results

**Two-stage pipeline vs. single stage:**

| Method | CD-L1 | CD-L2 | EMD | MSE |
| --- | ---: | ---: | ---: | ---: |
| CMDen-Net only | 19.97 | 5.17 | 37.42 | – |
| SDFDiff-Net only (antagonist cond.) | 17.64 | 3.95 | 35.19 | 0.0027 |
| CMDen-Net + SDFDiff-Net | **13.92** | **2.31** | **30.22** | **0.0018** |

**SMN backbone comparison:**

| Backbone | CD-L1 | CD-L2 | EMD |
| --- | ---: | ---: | ---: |
| PointNet++ | 16.24 | 3.04 | 34.32 |
| CMLP | 15.49 | 2.84 | 33.09 |
| DGCNN | 15.07 | 2.79 | 32.26 |
| PointNeXt | 14.83 | 2.42 | 31.91 |
| Transformer | 14.43 | 2.49 | 31.17 |
| **SMN (Ours)** | **13.92** | **2.31** | **30.22** |

**Dual-branch geometric enhancement:**

| Branch design | CD-L1 | CD-L2 | EMD | MSE |
| --- | ---: | ---: | ---: | ---: |
| Single Branch (Coord only) | 17.32 | 3.69 | 35.84 | 0.0029 |
| Single Branch (Coord + Normal) | 16.41 | 3.27 | 34.57 | 0.0021 |
| **Dual Branch (Ours)** | **13.92** | **2.31** | **30.22** | **0.0018** |

**Order Index Prompting (OIP) ablation:**

| Prompting | CD-L1 | CD-L2 | EMD |
| --- | ---: | ---: | ---: |
| No-Prompt | 17.05 | 3.50 | 35.76 |
| Order Prompt (OP) | 16.12 | 3.16 | 34.38 |
| **OIP (Ours)** | **13.92** | **2.31** | **30.22** |

**SDF-based non-penetration loss:**

| Setting | CD-L1 | CD-L2 | EMD | Penetration (%) |
| --- | ---: | ---: | ---: | ---: |
| Without SDF loss | 16.26 | 3.19 | 34.62 | 10.8 |
| **With SDF loss (Ours)** | **13.92** | **2.31** | **30.22** | **4.9** |

**CGOC:**

| Setting | CD-L1 | CD-L2 | EMD | MSE |
| --- | ---: | ---: | ---: | ---: |
| Without CGOC | 18.49 | 3.83 | 33.09 | 0.0032 |
| **With CGOC (Ours)** | **13.92** | **2.31** | **30.22** | **0.0018** |

**Antagonist-teeth conditioning in SDFDiff-Net:**

| Setting | Crown CD-L1 | Crown CD-L2 | Crown EMD | Occlusal CD-L1 | Occlusal CD-L2 | Occlusal EMD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Without antagonist teeth | 14.22 | 2.78 | 31.33 | 18.62 | 5.34 | 38.07 |
| **With antagonist teeth (Ours)** | **13.92** | **2.31** | **30.22** | **16.24** | **3.13** | **35.27** |

Antagonist conditioning yields marginal whole-crown improvement but substantially reduces occlusal-proxy errors, confirming that it primarily benefits occlusal-surface reconstruction.

## Implementation Details

| Setting | Value |
| --- | --- |
| Framework | PyTorch |
| GPU | NVIDIA RTX 4090 (single GPU) |
| Optimizer | Adam, lr = 1×10⁻⁴ |
| Batch size | 16 |
| Epochs | 300 (both stages) |
| kNN neighborhood | k = 32 |
| Voxel grid | 64³ |
| OIP embedding dim | 128 |
| CD loss weights | αL = 0.2, αM = 0.4 |
| Top-T (CGOC) | 30% of points |
| SoftMax temperature τ | 0.1 |
| Loss weights | λcd = 0.5, λcgoc = 0.3, λsdf = 0.2 |
| VAE latent dimension | 512 |
| KL weight β | 1×10⁻⁵ |
| DDPM timesteps T | 800 |
| Marching Cubes grid | 128³ |

## Repository Status

The repository contains a full project scaffold for the two-stage pipeline:

- Teeth3DS+ dataset loader for `upper_teeth.txt / upper_label.txt / lower_teeth.txt / lower_label.txt`
- Stage-1 `CMDen-Net` with tooth-wise FPS, dual-branch geometric enhancement, SMN-style serialization, and point pyramid decoding
- Stage-1 losses: multi-resolution Chamfer distance, CGOC, and non-penetration
- Stage-2 latent SDF VAE, DDPM scheduler, FiLM-conditioned Mamba denoiser, and mesh extraction path
- Training scripts for stage 1 and stage 2, plus mesh inference

This is a faithful engineering reproduction of the method described in the paper. It is intended as a strong implementation baseline and may still require further tuning to exactly match the reported metrics.

## Installation

Create an environment and install the dependencies listed in [requirements.txt](requirements.txt).

```bash
pip install -r requirements.txt
```

## Code Structure

- `cacrnet/data`: Teeth3DS+ dataset parsing and case assembly
- `cacrnet/models`: CMDen-Net, latent SDF VAE, SDFDiff-Net, and the integrated CACR-Net wrapper
- `cacrnet/losses`: Chamfer, CGOC, and non-penetration losses
- `cacrnet/diffusion`: DDPM scheduler and timestep embeddings
- `scripts`: stage-1 training, stage-2 training, and inference entry points
- `configs`: default YAML configs for training and inference

## Usage

Stage 1:

```bash
python scripts/train_stage1.py --config configs/stage1.yaml
```

Stage 2:

```bash
python scripts/train_stage2.py --config configs/stage2.yaml --stage1_ckpt outputs/stage1/stage1_epoch_300.pt
```

Inference:

```bash
python scripts/infer_mesh.py \
  --config configs/infer.yaml \
  --stage1_ckpt outputs/stage1/stage1_epoch_300.pt \
  --vae_ckpt outputs/stage2/vae_epoch_300.pt \
  --diff_ckpt outputs/stage2/diff_epoch_300.pt \
  --output outputs/infer/sample.obj
```

## Citation

If you find this repository useful, please cite the paper below. The BibTeX entry will be updated once publication details are available.

```bibtex
@misc{cacrnet2026,
  title = {CACR-Net: Constraint-Aware Occlusion-Guided Dental Crown Reconstruction with Latent SDF Diffusion},
  year = {2026},
  note = {Manuscript under review}
}
```

## Acknowledgements

- [Teeth3DS / Teeth3DS+](https://osf.io/xctdy/)
- Prior reconstruction baselines: PF-Net, FSC, AdaPoinTr, and PDCNet
- Reference repositories used for implementation guidance: PointCloudMamba, PF-Net, and Diffusion-SDF
