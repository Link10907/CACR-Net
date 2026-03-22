# CACR-Net

Official repository for **CACR-Net: Constraint-Aware Occlusion-Guided Dental Crown Reconstruction with Latent SDF Diffusion**.

CACR-Net is a two-stage framework for reconstructing anatomically accurate dental crowns from intraoral scans. It combines an occlusion-aware point-cloud reconstruction network with latent signed distance field (SDF) diffusion to generate closed crown meshes suitable for CAD/CAM workflows.

## Method

**Stage 1 — CMDen-Net** encodes a single-arch intraoral scan using a Serialized-Mamba Network (SMN) with four space-filling curve serializations (Hilbert, Z-order, raster, zigzag) and dual-branch geometric enhancement. A hierarchical point pyramid decoder produces an initial crown point cloud at three resolution levels. Training is supervised by multi-resolution Chamfer distance, a Curvature-Guided Occlusal Constraint (CGOC), and an SDF-based non-penetration loss.

**Stage 2 — SDFDiff-Net** refines the reconstruction in a learned latent SDF space. A VAE encodes ground-truth crown SDFs into a compact latent code, and a conditional DDPM denoiser — built with FiLM-modulated Mamba layers — iteratively recovers the latent code conditioned on the stage-1 output and antagonist teeth. Marching Cubes extracts the final closed triangular mesh.

## Installation

```bash
pip install -r requirements.txt
```

## Code Structure

```
cacrnet/
  data/        Teeth3DS+ dataset loader
  models/      CMDen-Net, SDF VAE, SDFDiff-Net, CACR-Net wrapper
  losses/      Chamfer, CGOC, non-penetration losses
  diffusion/   DDPM scheduler and timestep embeddings
  utils/       Point cloud and SDF utilities
  engine/      Training and inference logic
scripts/       Training and inference entry points
configs/       Default YAML configurations
tools/         Data preprocessing and visualization
```

## Usage

```bash
# Stage 1
python scripts/train_stage1.py --config configs/stage1.yaml

# Stage 2
python scripts/train_stage2.py --config configs/stage2.yaml \
  --stage1_ckpt outputs/stage1/stage1_epoch_300.pt

# Inference
python scripts/infer_mesh.py \
  --config configs/infer.yaml \
  --stage1_ckpt outputs/stage1/stage1_epoch_300.pt \
  --vae_ckpt outputs/stage2/vae_epoch_300.pt \
  --diff_ckpt outputs/stage2/diff_epoch_300.pt \
  --output outputs/infer/sample.obj
```

## Citation

```bibtex
@misc{cacrnet2026,
  title = {CACR-Net: Constraint-Aware Occlusion-Guided Dental Crown Reconstruction with Latent SDF Diffusion},
  year = {2026},
  note = {Manuscript under review}
}
```

## Acknowledgements

- [Teeth3DS / Teeth3DS+](https://osf.io/xctdy/)
- Reference implementations: PointCloudMamba, PF-Net, Diffusion-SDF
