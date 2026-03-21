from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from cacrnet.utils.pointcloud import estimate_normals_open3d, normalize_point_cloud


def _read_points(path: str) -> np.ndarray:
    points = np.loadtxt(path, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return points


def _read_labels(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as handle:
        values = [int(line.strip()) for line in handle if line.strip()]
    return np.asarray(values, dtype=np.int32)


def _discover_cases(root: str) -> List[str]:
    required = ("upper_teeth.txt", "upper_label.txt", "lower_teeth.txt", "lower_label.txt")
    candidates = []
    if all(os.path.exists(os.path.join(root, name)) for name in required):
        return [root]

    for item in sorted(os.listdir(root)):
        case_dir = os.path.join(root, item)
        if not os.path.isdir(case_dir):
            continue
        if all(os.path.exists(os.path.join(case_dir, name)) for name in required):
            candidates.append(case_dir)
    if not candidates:
        raise FileNotFoundError(f"No Teeth3DS+ cases found under {root}")
    return candidates


@dataclass
class JawData:
    blocks: np.ndarray
    labels: np.ndarray
    valid_blocks: List[int]
    centroid: np.ndarray
    scale: float


class Teeth3DSPlusDataset(Dataset):
    def __init__(
        self,
        root: str,
        points_per_tooth: int = 1024,
        teeth_per_jaw: int = 16,
        include_normals: bool = True,
        normal_radius: float = 0.1,
        normal_max_nn: int = 30,
    ) -> None:
        super().__init__()
        self.root = root
        self.points_per_tooth = points_per_tooth
        self.teeth_per_jaw = teeth_per_jaw
        self.include_normals = include_normals
        self.normal_radius = normal_radius
        self.normal_max_nn = normal_max_nn

        self.case_dirs = _discover_cases(root)
        self.cache: List[Dict[str, JawData]] = []
        self.index: List[Tuple[int, int, int]] = []

        for case_idx, case_dir in enumerate(self.case_dirs):
            upper = self._load_jaw(case_dir, "upper")
            lower = self._load_jaw(case_dir, "lower")
            self.cache.append({"upper": upper, "lower": lower})
            for block_idx in upper.valid_blocks:
                self.index.append((case_idx, 0, block_idx))
            for block_idx in lower.valid_blocks:
                self.index.append((case_idx, 1, block_idx))

    def _load_jaw(self, case_dir: str, jaw: str) -> JawData:
        point_path = os.path.join(case_dir, f"{jaw}_teeth.txt")
        label_path = os.path.join(case_dir, f"{jaw}_label.txt")
        points = _read_points(point_path)
        labels = _read_labels(label_path)

        xyz = points[:, :3]
        if labels.shape[0] * self.points_per_tooth != xyz.shape[0]:
            raise ValueError(
                f"{case_dir} {jaw}: labels={labels.shape[0]}, points={xyz.shape[0]}, "
                f"points_per_tooth={self.points_per_tooth}"
            )

        xyz, centroid, scale = normalize_point_cloud(xyz)
        blocks_xyz = xyz.reshape(labels.shape[0], self.points_per_tooth, 3)

        if blocks_xyz.shape[0] < self.teeth_per_jaw:
            pad = self.teeth_per_jaw - blocks_xyz.shape[0]
            blocks_xyz = np.concatenate(
                [blocks_xyz, np.zeros((pad, self.points_per_tooth, 3), dtype=np.float32)],
                axis=0,
            )
            labels = np.concatenate([labels, np.zeros((pad,), dtype=np.int32)], axis=0)

        if self.include_normals:
            flat_xyz = blocks_xyz.reshape(-1, 3)
            nonzero = np.any(flat_xyz != 0, axis=1)
            normals = np.zeros_like(flat_xyz, dtype=np.float32)
            if np.any(nonzero):
                normals[nonzero] = estimate_normals_open3d(
                    flat_xyz[nonzero],
                    radius=self.normal_radius,
                    max_nn=self.normal_max_nn,
                )
            blocks = np.concatenate([blocks_xyz, normals.reshape(blocks_xyz.shape)], axis=-1)
        else:
            blocks = blocks_xyz

        valid = []
        for tooth_idx, tooth_id in enumerate(labels):
            if int(tooth_id) == 0:
                continue
            if np.allclose(blocks[tooth_idx, :, :3], 0.0):
                continue
            valid.append(tooth_idx)

        return JawData(
            blocks=blocks.astype(np.float32),
            labels=labels.astype(np.int32),
            valid_blocks=valid,
            centroid=centroid,
            scale=scale,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_idx, jaw_flag, tooth_block_idx = self.index[idx]
        jaw_key = "upper" if jaw_flag == 0 else "lower"
        other_key = "lower" if jaw_key == "upper" else "upper"

        jaw_data = self.cache[case_idx][jaw_key]
        antagonist_data = self.cache[case_idx][other_key]

        target_points = jaw_data.blocks[tooth_block_idx].copy()
        masked_arch = jaw_data.blocks.copy()
        masked_arch[tooth_block_idx] = 0.0

        env_points = np.concatenate(
            [
                masked_arch.reshape(-1, masked_arch.shape[-1]),
                antagonist_data.blocks.reshape(-1, antagonist_data.blocks.shape[-1]),
            ],
            axis=0,
        )

        return {
            "arch": torch.from_numpy(masked_arch.reshape(-1, masked_arch.shape[-1])).float(),
            "arch_blocks": torch.from_numpy(masked_arch).float(),
            "full_arch": torch.from_numpy(jaw_data.blocks.reshape(-1, jaw_data.blocks.shape[-1])).float(),
            "target_points": torch.from_numpy(target_points).float(),
            "target_xyz": torch.from_numpy(target_points[:, :3]).float(),
            "antagonist": torch.from_numpy(
                antagonist_data.blocks.reshape(-1, antagonist_data.blocks.shape[-1])
            ).float(),
            "environment": torch.from_numpy(env_points).float(),
            "case_idx": torch.tensor(case_idx, dtype=torch.long),
            "jaw_flag": torch.tensor(jaw_flag, dtype=torch.long),
            "tooth_block_idx": torch.tensor(tooth_block_idx, dtype=torch.long),
            "tooth_id": torch.tensor(int(jaw_data.labels[tooth_block_idx]), dtype=torch.long),
            "centroid": torch.from_numpy(jaw_data.centroid).float(),
            "scale": torch.tensor(jaw_data.scale, dtype=torch.float32),
        }
