from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cacrnet.config import load_config
from cacrnet.data import Teeth3DSPlusDataset
from cacrnet.engine.stage1 import build_stage1_model, compute_stage1_losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CMDen-Net for CACR-Net.")
    parser.add_argument("--config", type=str, default="configs/stage1.yaml")
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
        batch_size=project_cfg.stage1.batch_size,
        shuffle=True,
        num_workers=project_cfg.stage1.num_workers,
    )

    device = torch.device(project_cfg.runtime.device)
    model = build_stage1_model(project_cfg.stage1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=project_cfg.stage1.learning_rate,
        weight_decay=project_cfg.stage1.weight_decay,
    )

    output_dir = Path(project_cfg.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(project_cfg.stage1.epochs):
        progress = tqdm(loader, desc=f"stage1 epoch {epoch + 1}/{project_cfg.stage1.epochs}")
        for batch in progress:
            batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
            stage1_out = model(batch["arch"], batch["arch_blocks"])
            losses = compute_stage1_losses(stage1_out, batch, project_cfg.stage1)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            progress.set_postfix(loss=float(losses["loss"].detach().cpu()))

        if (epoch + 1) % project_cfg.runtime.checkpoint_interval == 0:
            torch.save(model.state_dict(), output_dir / f"stage1_epoch_{epoch + 1:03d}.pt")


if __name__ == "__main__":
    main()
