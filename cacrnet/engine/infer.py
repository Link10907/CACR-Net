from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from cacrnet.models.cacr_net import CACRNet


def save_obj(vertices: np.ndarray, faces: np.ndarray, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def run_inference(model: CACRNet, batch: Dict, output_path: str, grid_resolution: int = 128) -> Dict[str, np.ndarray]:
    mesh = model.reconstruct_mesh(batch, grid_resolution=grid_resolution)
    save_obj(mesh["vertices"], mesh["faces"], output_path)
    return mesh
