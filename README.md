# CACR-Net

Official repository for the paper **CACR-Net: Constraint-Aware Occlusion-Guided Dental Crown Reconstruction with Latent SDF Diffusion**.

CACR-Net is a two-stage framework for reconstructing anatomically accurate and functionally plausible dental crowns from intraoral scans. It combines an occlusion-aware point-cloud reconstruction network with latent signed distance field diffusion to generate closed crown meshes that can be integrated into CAD/CAM workflows.

## Overview

Dental crown reconstruction still faces three practical challenges:

- limited methods that generalize across all tooth types
- insufficient 3D modeling fidelity from intraoral scans
- lack of explicit constraints on occlusal morphology and non-penetration

CACR-Net addresses these issues with two stages:

1. **CMDen-Net** predicts an initial crown point cloud from a single-arch intraoral scan.
2. **SDFDiff-Net** refines the reconstruction in a latent SDF space and decodes a closed triangular crown mesh.

## Method

### Stage 1: CMDen-Net

- Tooth-wise Iterative Farthest Point Sampling (TIFPS) builds multi-resolution tooth-aware point sets.
- A Serialized-Mamba Network (SMN) models long-range geometric dependencies across the full arch.
- Dual-branch geometric enhancement separately encodes coordinates and normals.
- A Curvature-Guided Occlusal Constraint (CGOC) emphasizes cusps, grooves, and fossae.
- An SDF-based non-penetration loss suppresses collisions with neighboring dentition and antagonist teeth.

### Stage 2: SDFDiff-Net

- A variational autoencoder learns a decodable latent signed distance field space.
- Conditional diffusion is guided by the stage-1 predicted crown point cloud and antagonist teeth.
- The denoised latent code is decoded into a continuous SDF.
- Marching Cubes extracts the zero-level set as a CAD/CAM-ready closed mesh.

## Main Contributions

- A two-stage occlusion-guided framework for dental crown reconstruction across all tooth types.
- A Serialized-Mamba point-cloud encoder with order index prompting for efficient multi-tooth geometric modeling.
- Explicit occlusal and collision constraints through CGOC and SDF-based non-penetration supervision.
- A latent conditional SDF diffusion stage for generating closed crown meshes directly usable in CAD/CAM.

## Results

### Teeth3DS+

Experiments were conducted on the public [Teeth3DS+ dataset](https://osf.io/xctdy/), which contains 1,800 intraoral scans from 900 patients. The dataset was split at the patient level into training, validation, and testing sets with a 70/10/20 ratio.

After denormalization, CACR-Net reports average reconstruction errors of **0.7392 mm (CD-L1)** and **0.5923 mm (CD-L2)**.

| Method | CD-L1 | CD-L2 | EMD | MSE |
| --- | ---: | ---: | ---: | ---: |
| PF-Net | 26.64 | 7.64 | 41.36 | - |
| FSC | 20.49 | 5.51 | 38.92 | - |
| AdaPoinTr | 19.11 | 4.80 | 36.29 | - |
| PDCNet* | 54.39 | 8.41 | 75.31 | 0.0023 |
| **CACR-Net** | **13.92** | **2.31** | **30.22** | **0.0018** |

\* PDCNet values are reported in the paper as reference only because its official code was not available for re-evaluation under the same protocol.

Key ablation findings:

- The full `CMDen-Net + SDFDiff-Net` pipeline outperformed either stage used alone.
- SMN outperformed PointNet++, CMLP, DGCNN, PointNeXt, and Transformer backbones.
- The SDF loss reduced penetration rate from `10.8%` to `4.9%`.
- CGOC reduced occlusal-region reconstruction error and improved global metrics.
- Antagonist-teeth conditioning improved occlusal-proxy performance more than whole-crown performance.

## Repository Status

This repository is currently being organized. The following components will be released progressively:

- [ ] training code
- [ ] inference code
- [ ] pretrained checkpoints
- [ ] data preprocessing scripts
- [ ] evaluation scripts
- [ ] visualization utilities

Code and trained models will be released after paper acceptance.

## Citation

If you find this repository useful, please cite the paper below. The BibTeX entry can be updated once publication details are available.

```bibtex
@misc{cacrnet2026,
  title = {CACR-Net: Constraint-Aware Occlusion-Guided Dental Crown Reconstruction with Latent SDF Diffusion},
  year = {2026},
  note = {Manuscript under review}
}
```

## Acknowledgements

- [Teeth3DS / Teeth3DS+](https://osf.io/xctdy/)
- Prior reconstruction baselines including PF-Net, FSC, AdaPoinTr, and PDCNet
