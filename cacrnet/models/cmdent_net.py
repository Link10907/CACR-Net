from __future__ import annotations

import math
from typing import Dict, Iterable

import torch
from torch import nn

from cacrnet.models.common.mamba import ResidualMambaBlock
from cacrnet.models.common.serializers import serialize_sequence
from cacrnet.utils.pointcloud import toothwise_fps


def _knn_features(xyz: torch.Tensor, k: int) -> torch.Tensor:
    dist = torch.cdist(xyz, xyz)
    idx = dist.topk(k=min(k + 1, xyz.shape[1]), largest=False).indices[..., 1:]
    neighbors = torch.gather(
        xyz.unsqueeze(1).expand(-1, xyz.shape[1], -1, -1),
        2,
        idx.unsqueeze(-1).expand(-1, -1, -1, xyz.shape[-1]),
    )
    center = xyz.unsqueeze(2)
    return neighbors - center


class DualBranchGeometricEnhancer(nn.Module):
    def __init__(self, k_neighbors: int, hidden_dim: int):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normal_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points[..., :3]
        normals = points[..., 3:6]

        rel_xyz = _knn_features(xyz, self.k_neighbors)
        coord_feat = self.coord_mlp(rel_xyz).max(dim=2).values

        dist = torch.cdist(xyz, xyz)
        idx = dist.topk(k=min(self.k_neighbors + 1, xyz.shape[1]), largest=False).indices[..., 1:]
        neigh_normals = torch.gather(
            normals.unsqueeze(1).expand(-1, normals.shape[1], -1, -1),
            2,
            idx.unsqueeze(-1).expand(-1, -1, -1, normals.shape[-1]),
        )
        center_normals = normals.unsqueeze(2).expand_as(neigh_normals)
        variation = 1.0 - (center_normals * neigh_normals).sum(dim=-1, keepdim=True)
        normal_feat = self.normal_mlp(torch.cat([neigh_normals, variation], dim=-1)).max(dim=2).values
        return self.out_proj(torch.cat([coord_feat, normal_feat], dim=-1))


class SerializedMambaEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        global_dim: int,
        schemes: Iterable[str],
        voxel_resolution: int,
    ):
        super().__init__()
        self.schemes = tuple(schemes)
        self.voxel_resolution = voxel_resolution
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.scheme_embed = nn.Embedding(len(self.schemes), hidden_dim)
        self.blocks = nn.ModuleList([ResidualMambaBlock(hidden_dim) for _ in self.schemes])
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim * 2, global_dim),
            nn.GELU(),
            nn.Linear(global_dim, global_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.input_proj(points)
        xyz = points[..., :3]
        for scheme_idx, (scheme, block) in enumerate(zip(self.schemes, self.blocks)):
            order, inverse = serialize_sequence(xyz, scheme=scheme, resolution=self.voxel_resolution)
            gather_idx = order.unsqueeze(-1).expand(-1, -1, features.shape[-1])
            ordered = torch.gather(features, 1, gather_idx)
            prompt = self.scheme_embed.weight[scheme_idx].view(1, 1, -1).expand(features.shape[0], 2, -1)
            ordered = torch.cat([prompt[:, :1], ordered, prompt[:, 1:]], dim=1)
            ordered = block(ordered)
            ordered = ordered[:, 1:-1]
            scatter_idx = inverse.unsqueeze(-1).expand(-1, -1, ordered.shape[-1])
            features = torch.gather(ordered, 1, scatter_idx)

        pooled = torch.cat([features.max(dim=1).values, features.mean(dim=1)], dim=-1)
        return self.out_proj(pooled)


class PointPyramidDecoder(nn.Module):
    def __init__(self, global_dim: int, low_points: int, mid_points: int, high_points: int):
        super().__init__()
        self.low_points = low_points
        self.mid_points = mid_points
        self.high_points = high_points
        self.low_head = nn.Linear(global_dim, low_points * 3)
        self.mid_head = nn.Linear(global_dim, mid_points * 3)
        self.high_head = nn.Linear(global_dim, high_points * 3)

    @staticmethod
    def _expand(points: torch.Tensor, out_points: int) -> torch.Tensor:
        repeat = math.ceil(out_points / points.shape[1])
        expanded = points.repeat_interleave(repeat, dim=1)
        return expanded[:, :out_points]

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor) -> Dict[str, torch.Tensor]:
        pl = self.low_head(f3).view(f3.shape[0], self.low_points, 3)
        pm_delta = self.mid_head(f2).view(f2.shape[0], self.mid_points, 3)
        pm = self._expand(pl, self.mid_points) + pm_delta
        ph_delta = self.high_head(f1).view(f1.shape[0], self.high_points, 3)
        ph = self._expand(pm, self.high_points) + ph_delta
        return {"low": pl, "mid": pm, "high": ph}


class CMDenNet(nn.Module):
    def __init__(
        self,
        point_dim: int = 6,
        hidden_dim: int = 128,
        global_dim: int = 512,
        voxel_resolution: int = 64,
        k_neighbors: int = 32,
        low_points: int = 256,
        mid_points: int = 512,
        high_points: int = 1024,
        schemes: Iterable[str] = ("hilbert", "z_order", "raster", "zigzag"),
    ):
        super().__init__()
        self.low_points = low_points
        self.mid_points = mid_points
        self.high_points = high_points
        self.k_neighbors = k_neighbors

        self.enhancer = DualBranchGeometricEnhancer(k_neighbors=k_neighbors, hidden_dim=hidden_dim)
        self.low_encoder = SerializedMambaEncoder(hidden_dim + point_dim, hidden_dim, global_dim, schemes, voxel_resolution)
        self.mid_encoder = SerializedMambaEncoder(hidden_dim + point_dim, hidden_dim, global_dim, schemes, voxel_resolution)
        self.high_encoder = SerializedMambaEncoder(hidden_dim + point_dim, hidden_dim, global_dim, schemes, voxel_resolution)
        self.global_to_f1 = nn.Sequential(nn.Linear(global_dim * 3, global_dim), nn.GELU(), nn.Linear(global_dim, global_dim))
        self.global_to_f2 = nn.Sequential(nn.Linear(global_dim * 3, global_dim), nn.GELU(), nn.Linear(global_dim, global_dim))
        self.global_to_f3 = nn.Sequential(nn.Linear(global_dim * 3, global_dim), nn.GELU(), nn.Linear(global_dim, global_dim))
        self.decoder = PointPyramidDecoder(global_dim, low_points, mid_points, high_points)

    def _encode_resolution(self, points: torch.Tensor, blocks: torch.Tensor, samples: int, encoder: SerializedMambaEncoder) -> torch.Tensor:
        sampled = toothwise_fps(blocks, samples)
        sampled = sampled.reshape(sampled.shape[0], -1, sampled.shape[-1])
        enhanced = self.enhancer(sampled)
        return encoder(torch.cat([sampled, enhanced], dim=-1))

    def forward(self, arch_points: torch.Tensor, arch_blocks: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_low = self._encode_resolution(arch_points, arch_blocks, self.low_points, self.low_encoder)
        feat_mid = self._encode_resolution(arch_points, arch_blocks, self.mid_points, self.mid_encoder)
        feat_high = self._encode_resolution(arch_points, arch_blocks, self.high_points, self.high_encoder)
        global_feat = torch.cat([feat_low, feat_mid, feat_high], dim=-1)
        f1 = self.global_to_f1(global_feat)
        f2 = self.global_to_f2(global_feat)
        f3 = self.global_to_f3(global_feat)
        pred = self.decoder(f1, f2, f3)
        pred["global_feature"] = global_feat
        return pred
