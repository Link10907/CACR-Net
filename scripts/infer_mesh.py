from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cacrnet.config import load_config
from cacrnet.data import Teeth3DSPlusDataset
from cacrnet.engine.infer import run_inference
from cacrnet.engine.stage1 import build_stage1_model
from cacrnet.engine.stage2 import build_stage2_components
from cacrnet.models.cacr_net import CACRNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct dental crown meshes with CACR-Net.")
    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--diff_ckpt", type=str, required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--output", type=str, default="outputs/infer/sample.obj")
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
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    iterator = iter(loader)
    for _ in range(args.sample_index + 1):
        batch = next(iterator)

    device = torch.device(project_cfg.runtime.device)

    # Build models
    stage1 = build_stage1_model(project_cfg.stage1)
    vae, denoiser, scheduler = build_stage2_components(project_cfg.stage2)

    # Load checkpoints
    stage1.load_state_dict(torch.load(args.stage1_ckpt, map_location=device, weights_only=True))
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device, weights_only=True))
    denoiser.load_state_dict(torch.load(args.diff_ckpt, map_location=device, weights_only=True))

    # Assemble full pipeline
    model = CACRNet(stage1, vae, denoiser, scheduler).to(device).eval()

    batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = run_inference(model, batch, str(output_path), grid_resolution=project_cfg.stage2.grid_resolution)
    print(f"Saved mesh to {output_path}: {mesh['vertices'].shape[0]} vertices, {mesh['faces'].shape[0]} faces")


if __name__ == "__main__":
    main()
