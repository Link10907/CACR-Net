# Project Layout

`cacrnet/data`
Dataset adapters for Teeth3DS+ style `upper_teeth.txt` and `lower_teeth.txt` cases.

`cacrnet/models`
Stage-1 CMDen-Net, stage-2 latent SDF VAE, and SDFDiff-Net denoiser.

`cacrnet/losses`
Chamfer, CGOC, and non-penetration losses.

`cacrnet/diffusion`
Minimal DDPM scheduler and time embeddings.

`scripts`
Training and inference entry points.
