from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cacrnet.config import load_config
from cacrnet.data import Teeth3DSPlusDataset
from cacrnet.engine.stage1 import build_stage1_model
from cacrnet.engine.stage2 import build_stage2_components, compute_stage2_losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train latent SDF diffusion for CACR-Net.")
    parser.add_argument("--config", type=str, default="configs/stage2.yaml")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    args = parser.parse_args()

    project_cfg = load_config(args.config)
    dataset = Teeth3DSPlusDataset(
        root=project_cfg.dataset.root,
        points_per_tooth=project_cfg.dataset.points_per_tooth,
        teeth_per_jaw=project_cfg.dataset.teeth_per_jaw,
        include_normals=project_cfg.dataset.include_normals,
        normal_radius=project_cfg.dataset.normal_radius,
        normal_max_nn=project_cfg.dataset.normal_max_nn,
    )
    loader = DataLoader(
        dataset,
        batch_size=project_cfg.stage2.batch_size,
        shuffle=True,
        num_workers=project_cfg.stage2.num_workers,
    )

    device = torch.device(project_cfg.runtime.device)
    stage1 = build_stage1_model(project_cfg.stage1).to(device)
    stage1.load_state_dict(torch.load(args.stage1_ckpt, map_location=device))
    stage1.eval()
    for param in stage1.parameters():
        param.requires_grad_(False)

    vae, denoiser, _scheduler = build_stage2_components(project_cfg.stage2)
    vae = vae.to(device)
    denoiser = denoiser.to(device)
    optimizer = torch.optim.AdamW(
        list(vae.parameters()) + list(denoiser.parameters()),
        lr=project_cfg.stage2.learning_rate,
        weight_decay=project_cfg.stage2.weight_decay,
    )

    output_dir = Path(project_cfg.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(project_cfg.stage2.epochs):
        progress = tqdm(loader, desc=f"stage2 epoch {epoch + 1}/{project_cfg.stage2.epochs}")
        for batch in progress:
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            with torch.no_grad():
                stage1_out = stage1(batch["arch"], batch["arch_blocks"])
            losses = compute_stage2_losses(vae, denoiser, _scheduler, stage1_out, batch, project_cfg.stage2)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            progress.set_postfix(loss=float(losses["loss"].detach().cpu()))

        if (epoch + 1) % project_cfg.runtime.checkpoint_interval == 0:
            torch.save(vae.state_dict(), output_dir / f"vae_epoch_{epoch + 1:03d}.pt")
            torch.save(denoiser.state_dict(), output_dir / f"diff_epoch_{epoch + 1:03d}.pt")


if __name__ == "__main__":
    main()
