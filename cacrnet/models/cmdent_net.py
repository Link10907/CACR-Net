from __future__ import annotations

import math
import random
from typing import Dict, Iterable

import torch
from torch import nn

from cacrnet.models.common.mamba import BidirectionalMambaBlock
from cacrnet.models.common.serializers import serialize_sequence
from cacrnet.utils.pointcloud import toothwise_fps


class DualBranchGeometricEnhancer(nn.Module):
    """Dual-branch geometric enhancement (paper Sec III-A-2).

    Coordinate branch: pointwise MLP on neighbor coordinates + max-pool.
    Normal branch: normal-variation score s_j = |1 - <n_i, n_j>|, mapped by
                   pointwise MLP on (n_j, s_j) + max-pool.
    Outputs are fused at the point level.
    """

    def __init__(self, k_neighbors: int, hidden_dim: int):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normal_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),  # (n_j: 3, s_j: 1)
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

        # Build kNN once and reuse for both branches
        dist = torch.cdist(xyz, xyz)
        k_actual = min(self.k_neighbors + 1, xyz.shape[1])
        knn_idx = dist.topk(k=k_actual, largest=False).indices[..., 1:]  # exclude self

        # Coordinate branch: relative coordinates -> MLP -> max-pool
        neighbors_xyz = torch.gather(
            xyz.unsqueeze(1).expand(-1, xyz.shape[1], -1, -1),
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        rel_xyz = neighbors_xyz - xyz.unsqueeze(2)
        coord_feat = self.coord_mlp(rel_xyz).max(dim=2).values

        # Normal branch: (n_j, s_j) -> MLP -> max-pool
        neigh_normals = torch.gather(
            normals.unsqueeze(1).expand(-1, normals.shape[1], -1, -1),
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        center_normals = normals.unsqueeze(2).expand_as(neigh_normals)
        # s_j = |1 - <n_i, n_j>|  (paper Sec III-A-2)
        variation = torch.abs(1.0 - (center_normals * neigh_normals).sum(dim=-1, keepdim=True))
        normal_feat = self.normal_mlp(torch.cat([neigh_normals, variation], dim=-1)).max(dim=2).values

        return self.out_proj(torch.cat([coord_feat, normal_feat], dim=-1))


class SerializedMambaEncoder(nn.Module):
    """Serialized-Mamba Network (SMN) backbone (paper Sec III-A-2, Fig. 2).

    Key design decisions from the paper:
    - Four Mamba blocks, each randomly selects a serialization scheme during training.
    - Order Index Prompt (OIP): serialization position indices are mapped by a
      shared linear layer; two OIP tokens are inserted at sequence boundaries.
    - Bidirectional SSMs in each Mamba block.
    - Residual connections between blocks with channel alignment if needed.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        global_dim: int,
        schemes: Iterable[str] = ("hilbert", "z_order", "raster", "zigzag"),
        voxel_resolution: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.schemes = tuple(schemes)
        self.voxel_resolution = voxel_resolution
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # OIP: shared linear layer maps serialization position index to prompt embedding
        self.oip_embed = nn.Linear(1, hidden_dim)
        # Mamba blocks
        self.blocks = nn.ModuleList([BidirectionalMambaBlock(hidden_dim) for _ in range(num_blocks)])
        # Output projection: concat max-pool and mean-pool
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, global_dim),
            nn.GELU(),
            nn.Linear(global_dim, global_dim),
        )

    def _make_oip(self, seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create two OIP tokens from boundary serialization indices."""
        # Map the first (0) and last (seq_len-1) position indices
        idx_start = torch.zeros(batch_size, 1, 1, device=device)
        idx_end = torch.full((batch_size, 1, 1), (seq_len - 1) / max(seq_len - 1, 1), device=device)
        oip_start = self.oip_embed(idx_start)  # (B, 1, hidden_dim)
        oip_end = self.oip_embed(idx_end)       # (B, 1, hidden_dim)
        return oip_start, oip_end

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.input_proj(points)
        xyz = points[..., :3]
        B, N, D = features.shape

        for block in self.blocks:
            # Randomly select serialization scheme during training (paper Sec III-A-2)
            if self.training:
                scheme = random.choice(self.schemes)
            else:
                # Deterministic during eval: cycle through schemes
                scheme = self.schemes[0]

            order, inverse = serialize_sequence(xyz, scheme=scheme, resolution=self.voxel_resolution)
            gather_idx = order.unsqueeze(-1).expand(-1, -1, D)
            ordered = torch.gather(features, 1, gather_idx)

            # Insert OIP tokens at boundaries
            oip_start, oip_end = self._make_oip(N, B, features.device)
            ordered = torch.cat([oip_start, ordered, oip_end], dim=1)

            # Bidirectional Mamba block
            ordered = block(ordered)

            # Remove OIP tokens
            ordered = ordered[:, 1:-1]

            # Scatter back to original order
            scatter_idx = inverse.unsqueeze(-1).expand(-1, -1, ordered.shape[-1])
            features = torch.gather(ordered, 1, scatter_idx)

        features = self.out_norm(features)
        pooled = torch.cat([features.max(dim=1).values, features.mean(dim=1)], dim=-1)
        return self.out_proj(pooled)


class PointPyramidDecoder(nn.Module):
    """Hierarchical point pyramid decoder (paper Sec III-A-3).

    Inspired by FPN and Point Pyramid Decoder (PF-Net, ref [9]):
    - F3 -> P_L (low-res skeletal)
    - F2 -> delta, P_M = expand(P_L) + delta
    - F1 -> delta, P_H = expand(P_M) + delta
    """

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
        """Upsample by repeating points to match target count."""
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
    """Curvature-Guided Mamba Dental Crown Network (paper Sec III-A).

    Processes three resolution levels via separate SMN encoders with TIFPS,
    concatenates multi-resolution features into a global feature F, then
    decodes via hierarchical point pyramid.
    """

    def __init__(
        self,
        point_dim: int = 6,
        hidden_dim: int = 128,
        global_dim: int = 512,
        voxel_resolution: int = 64,
        k_neighbors: int = 32,
        low_points: int = 512,
        mid_points: int = 1024,
        high_points: int = 2048,
        schemes: Iterable[str] = ("hilbert", "z_order", "raster", "zigzag"),
    ):
        super().__init__()
        self.low_points = low_points
        self.mid_points = mid_points
        self.high_points = high_points
        self.k_neighbors = k_neighbors

        self.enhancer = DualBranchGeometricEnhancer(k_neighbors=k_neighbors, hidden_dim=hidden_dim)
        encoder_in_dim = hidden_dim + point_dim
        self.low_encoder = SerializedMambaEncoder(encoder_in_dim, hidden_dim, global_dim, schemes, voxel_resolution)
        self.mid_encoder = SerializedMambaEncoder(encoder_in_dim, hidden_dim, global_dim, schemes, voxel_resolution)
        self.high_encoder = SerializedMambaEncoder(encoder_in_dim, hidden_dim, global_dim, schemes, voxel_resolution)

        # F -> F1, F2, F3 via MLP (paper Sec III-A-3)
        self.global_to_f1 = nn.Sequential(nn.Linear(global_dim * 3, global_dim), nn.GELU(), nn.Linear(global_dim, global_dim))
        self.global_to_f2 = nn.Sequential(nn.Linear(global_dim * 3, global_dim), nn.GELU(), nn.Linear(global_dim, global_dim))
        self.global_to_f3 = nn.Sequential(nn.Linear(global_dim * 3, global_dim), nn.GELU(), nn.Linear(global_dim, global_dim))
        self.decoder = PointPyramidDecoder(global_dim, low_points, mid_points, high_points)

    def _encode_resolution(
        self, blocks: torch.Tensor, total_points: int, encoder: SerializedMambaEncoder
    ) -> torch.Tensor:
        """Encode arch at one resolution level.

        TIFPS samples (total_points / num_teeth) points per tooth, yielding
        total_points across the arch. This keeps the kNN in the enhancer
        tractable (e.g., 2048 total points → 2048×2048 distance matrix).
        """
        num_teeth = blocks.shape[1]
        per_tooth = max(1, total_points // num_teeth)
        sampled = toothwise_fps(blocks, per_tooth)
        sampled = sampled.reshape(sampled.shape[0], -1, sampled.shape[-1])
        enhanced = self.enhancer(sampled)
        return encoder(torch.cat([sampled, enhanced], dim=-1))

    def forward(self, arch_points: torch.Tensor, arch_blocks: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_low = self._encode_resolution(arch_blocks, self.low_points, self.low_encoder)
        feat_mid = self._encode_resolution(arch_blocks, self.mid_points, self.mid_encoder)
        feat_high = self._encode_resolution(arch_blocks, self.high_points, self.high_encoder)

        global_feat = torch.cat([feat_low, feat_mid, feat_high], dim=-1)
        f1 = self.global_to_f1(global_feat)
        f2 = self.global_to_f2(global_feat)
        f3 = self.global_to_f3(global_feat)

        pred = self.decoder(f1, f2, f3)
        pred["global_feature"] = global_feat
        return pred
