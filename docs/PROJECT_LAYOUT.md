# Project Layout

`cacrnet/data`
Dataset adapters for Teeth3DS+ style `upper_teeth.txt` and `lower_teeth.txt` cases.

`cacrnet/models`
Stage-1 CMDen-Net (Serialized-Mamba encoder + point pyramid decoder), stage-2 latent SDF VAE, and SDFDiff-Net (FiLM-modulated Mamba denoiser).

`cacrnet/models/common`
Shared building blocks: bidirectional Mamba blocks, PointNet encoder, and space-filling curve serializers (Hilbert, Z-order, raster, zigzag).

`cacrnet/losses`
Chamfer distance (multi-resolution, squared L2), CGOC (curvature-guided occlusal constraint), and SDF-based non-penetration losses.

`cacrnet/diffusion`
DDPM scheduler with standard posterior variance and sinusoidal timestep embeddings.

`cacrnet/utils`
Point cloud utilities (FPS, TIFPS, normalization, normal estimation) and SDF utilities (query sampling, mesh extraction via Marching Cubes).

`cacrnet/engine`
Training logic for stage 1 and stage 2, plus inference pipeline.

`scripts`
Training and inference entry points.

`tools`
Data preprocessing and visualization utilities.

`configs`
Default YAML configurations for stage 1, stage 2, and inference.
