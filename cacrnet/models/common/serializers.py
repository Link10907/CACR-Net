from __future__ import annotations

from typing import Iterable, Tuple

import torch

try:
    from hilbertcurve.hilbertcurve import HilbertCurve
except Exception:  # pragma: no cover - optional dependency
    HilbertCurve = None


def _voxelize(xyz: torch.Tensor, resolution: int) -> torch.Tensor:
    """Map normalized [0, 1] coordinates to integer voxel indices (Eq. 1).

    (x^v, y^v, z^v) = int((x^s, y^s, z^s) * V)
    """
    # Clamp to [0, 1] then map to grid
    coords = xyz.clamp(0.0, 1.0 - 1e-6)
    return (coords * resolution).long().clamp(min=0, max=resolution - 1)


def _morton_part1by2(n: torch.Tensor) -> torch.Tensor:
    n = n & 0x1FFFFF
    n = (n | (n << 32)) & 0x1F00000000FFFF
    n = (n | (n << 16)) & 0x1F0000FF0000FF
    n = (n | (n << 8)) & 0x100F00F00F00F00F
    n = (n | (n << 4)) & 0x10C30C30C30C30C3
    n = (n | (n << 2)) & 0x1249249249249249
    return n


def _z_order_index(grid_xyz: torch.Tensor) -> torch.Tensor:
    """Z-order (Morton) curve index."""
    x, y, z = grid_xyz.unbind(dim=-1)
    return _morton_part1by2(x) | (_morton_part1by2(y) << 1) | (_morton_part1by2(z) << 2)


def _raster_index(grid_xyz: torch.Tensor, resolution: int) -> torch.Tensor:
    """Row-major raster scan index."""
    x, y, z = grid_xyz.unbind(dim=-1)
    return x * resolution * resolution + y * resolution + z


def _zigzag_index(grid_xyz: torch.Tensor, resolution: int) -> torch.Tensor:
    """Zigzag (boustrophedon) scan index."""
    x, y, z = grid_xyz.unbind(dim=-1)
    zig_y = torch.where(x % 2 == 0, y, (resolution - 1) - y)
    zig_z = torch.where((x + y) % 2 == 0, z, (resolution - 1) - z)
    return x * resolution * resolution + zig_y * resolution + zig_z


def _hilbert_index_batch(grid_xyz: torch.Tensor, resolution: int) -> torch.Tensor:
    """Hilbert curve index, properly handling batched (B, N, 3) input."""
    if HilbertCurve is None:
        return _z_order_index(grid_xyz)

    bits = max(1, (resolution - 1).bit_length())
    curve = HilbertCurve(bits, 3)

    if grid_xyz.ndim == 2:
        # unbatched (N, 3)
        coords = grid_xyz.detach().cpu().tolist()
        distances = [curve.distance_from_point(list(map(int, c))) for c in coords]
        return torch.tensor(distances, device=grid_xyz.device, dtype=torch.long)
    else:
        # batched (B, N, 3)
        B, N, _ = grid_xyz.shape
        result = torch.empty(B, N, device=grid_xyz.device, dtype=torch.long)
        coords_np = grid_xyz.detach().cpu()
        for b in range(B):
            distances = [
                curve.distance_from_point(list(map(int, coords_np[b, i].tolist())))
                for i in range(N)
            ]
            result[b] = torch.tensor(distances, dtype=torch.long)
        return result


def serialize_sequence(
    xyz: torch.Tensor,
    scheme: str,
    resolution: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Serialize point cloud into 1-D sequence via space-filling curve.

    Args:
        xyz: (B, N, 3) point coordinates (expected in [0, 1] range)
        scheme: one of "hilbert", "z_order", "raster", "zigzag"
        resolution: voxel grid resolution V (paper uses 64^3)

    Returns:
        order: (B, N) sorting indices
        inverse: (B, N) inverse permutation to restore original order
    """
    grid_xyz = _voxelize(xyz, resolution)

    if scheme == "hilbert":
        order_key = _hilbert_index_batch(grid_xyz, resolution)
    elif scheme == "z_order":
        order_key = _z_order_index(grid_xyz)
    elif scheme == "raster":
        order_key = _raster_index(grid_xyz, resolution)
    elif scheme == "zigzag":
        order_key = _zigzag_index(grid_xyz, resolution)
    else:
        raise ValueError(f"Unsupported serialization scheme: {scheme}")

    order = torch.argsort(order_key, dim=-1)
    inverse = torch.argsort(order, dim=-1)
    return order, inverse


def supported_schemes() -> Iterable[str]:
    return ("hilbert", "z_order", "raster", "zigzag")
