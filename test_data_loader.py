import os
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import open3d as o3d
except Exception as e:
    raise ImportError("需要安装 open3d 用于法向量计算：pip install open3d") from e


def pc_normalize_xyz_ignore_zeros(pts: np.ndarray):
    """
    归一化（模仿你原来的 pc_normalize，忽略全0点）：
    - 只对 xyz 做平移+缩放
    - 返回归一化后的 xyz、centroid、scale
    """
    xyz = pts.copy().astype(np.float32)
    mask = np.any(xyz != 0, axis=1)
    if not np.any(mask):
        return xyz, np.zeros((3,), dtype=np.float32), 1.0

    valid = xyz[mask]
    centroid = np.mean(valid, axis=0)
    xyz[mask] = xyz[mask] - centroid

    m = np.max(np.sqrt(np.sum(xyz[mask] ** 2, axis=1)))
    m = float(m) if m > 1e-8 else 1.0
    xyz[mask] = xyz[mask] / m

    return xyz, centroid.astype(np.float32), m


def estimate_normals_open3d(points_xyz: np.ndarray, radius: float = 0.1, max_nn: int = 30):
    """
    给 (N,3) 点云估计法向量。
    - 默认按你旧代码：radius=0.1, max_nn=30
    - 会做“朝外”方向一致化：若法向量指向质心，则翻转
    """
    if points_xyz.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    centroid = np.mean(points_xyz, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

    normals = np.asarray(pcd.normals, dtype=np.float32)

    # 方向一致化：让法向量“远离质心”
    vec_to_centroid = centroid - points_xyz  # (N,3)
    dot = np.sum(normals * vec_to_centroid, axis=1)
    flip = dot > 0
    normals[flip] = -normals[flip]

    return normals


def _read_points_txt(path: str) -> np.ndarray:
    pts = np.loadtxt(path, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] < 3:
        raise ValueError(f"{path} 点云维度不足3，实际: {pts.shape}")
    return pts


def _read_label_txt(path: str) -> np.ndarray:
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(int(s))
    return np.array(ids, dtype=np.int32)


def _discover_case_dirs(root: str) -> List[str]:
    need = ["upper_teeth.txt", "upper_label.txt", "lower_teeth.txt", "lower_label.txt"]

    # root 本身就是一个case
    if all(os.path.exists(os.path.join(root, n)) for n in need):
        return [root]

    # root 下每个子目录一个case
    out = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        if all(os.path.exists(os.path.join(p, n)) for n in need):
            out.append(p)

    if not out:
        raise FileNotFoundError(
            f"在 {root} 没找到case数据。\n"
            f"需要每个case目录包含：upper_teeth.txt / upper_label.txt / lower_teeth.txt / lower_label.txt"
        )
    return out


def _is_all_zero_block(block_xyz: np.ndarray, eps: float = 0.0) -> bool:
    return np.all(np.abs(block_xyz) <= eps)


class Teeth3D(Dataset):
    """
    ✅ 适配你 test_PF.py 流程的测试 Dataset（不改网络/流程）：

    每个样本：
      ref       : [16384,6]  (整颌点云，删掉1颗牙后置0)
      ref_label : [1024,6]   (被删掉那颗牙，作为GT)
      meta      : [4] long   [case_idx, jaw_flag(0上1下), block_idx, tooth_id]

    一个case：
      14 上 + 14 下 = 28 个样本（自动忽略全0块/补0智齿）
    """

    def __init__(
        self,
        root: str,
        points_per_tooth: int = 1024,
        teeth_per_jaw: int = 16,
        zero_eps: float = 0.0,
        normalize: bool = True,
        normal_radius: float = 0.1,
        normal_max_nn: int = 30,
    ):
        super().__init__()
        self.root = root
        self.P = points_per_tooth
        self.T = teeth_per_jaw
        self.zero_eps = zero_eps
        self.normalize = normalize
        self.normal_radius = normal_radius
        self.normal_max_nn = normal_max_nn

        self.case_dirs = _discover_case_dirs(root)
        self.case_names = [os.path.basename(p.rstrip("\\/")) for p in self.case_dirs]

        # cache: 每个case缓存 upper/lower 的 blocks [16,1024,6] 和 ids [16]，以及valid块索引
        self.cache = []  # [{"upper":(blocks6, ids, valid), "lower":(...)}]
        # index: (case_idx, jaw_flag, block_idx)
        self.index: List[Tuple[int, int, int]] = []

        for case_idx, case_dir in enumerate(self.case_dirs):
            up_pts = _read_points_txt(os.path.join(case_dir, "upper_teeth.txt"))
            up_ids = _read_label_txt(os.path.join(case_dir, "upper_label.txt"))

            lo_pts = _read_points_txt(os.path.join(case_dir, "lower_teeth.txt"))
            lo_ids = _read_label_txt(os.path.join(case_dir, "lower_label.txt"))

            up_blocks6, up_ids, up_valid = self._prep_one_jaw(up_pts, up_ids, case_dir, "upper")
            lo_blocks6, lo_ids, lo_valid = self._prep_one_jaw(lo_pts, lo_ids, case_dir, "lower")

            self.cache.append({
                "upper": (up_blocks6, up_ids, up_valid),
                "lower": (lo_blocks6, lo_ids, lo_valid),
            })

            for bi in up_valid:
                self.index.append((case_idx, 0, bi))
            for bi in lo_valid:
                self.index.append((case_idx, 1, bi))

    def _prep_one_jaw(self, pts_any: np.ndarray, ids: np.ndarray, case_dir: str, jaw_name: str):
        """
        输入 pts_any: (N,3) 或 (N,>=3)，只取前三维xyz
        输出 blocks6: (16,1024,6)  (xyz + normals)
        """
        xyz = pts_any[:, 0:3].astype(np.float32)

        if len(ids) * self.P != xyz.shape[0]:
            raise ValueError(
                f"{case_dir} {jaw_name} 点数不匹配：label行数={len(ids)}，每牙{self.P} => "
                f"应为{len(ids)*self.P}点，实际{xyz.shape[0]}点"
            )

        old_T = len(ids)
        blocks_xyz = xyz.reshape(old_T, self.P, 3)

        # 补到16块保证16384点
        if old_T != self.T:
            pad_blocks = np.zeros((self.T - old_T, self.P, 3), dtype=np.float32)
            blocks_xyz = np.concatenate([blocks_xyz, pad_blocks], axis=0)

            pad_ids = np.zeros((self.T - old_T,), dtype=np.int32)  # tooth_id=0占位
            ids = np.concatenate([ids.astype(np.int32), pad_ids], axis=0)
            print(f"[WARN] {case_dir} {jaw_name} label不是{self.T}行，已补零到{self.T}块以匹配16384点。")
        else:
            ids = ids.astype(np.int32)

        # 归一化（忽略0点）
        if self.normalize:
            flat = blocks_xyz.reshape(self.T * self.P, 3)
            flat, _, _ = pc_normalize_xyz_ignore_zeros(flat)
            blocks_xyz = flat.reshape(self.T, self.P, 3)

        # 计算整颌 normals：只对非0点计算，0点保持0
        flat_xyz = blocks_xyz.reshape(self.T * self.P, 3)
        nonzero_mask = np.any(flat_xyz != 0, axis=1)
        normals_flat = np.zeros((flat_xyz.shape[0], 3), dtype=np.float32)

        if np.any(nonzero_mask):
            normals_valid = estimate_normals_open3d(
                flat_xyz[nonzero_mask],
                radius=self.normal_radius,
                max_nn=self.normal_max_nn
            )
            normals_flat[nonzero_mask] = normals_valid

        blocks_n = normals_flat.reshape(self.T, self.P, 3)
        blocks6 = np.concatenate([blocks_xyz, blocks_n], axis=2)  # [T,P,6]

        # 有效牙：label!=0 且 xyz块非全0
        valid = []
        for i in range(self.T):
            if int(ids[i]) == 0:
                continue
            if _is_all_zero_block(blocks6[i, :, 0:3], eps=self.zero_eps):
                continue
            valid.append(i)

        return blocks6.astype(np.float32), ids.astype(np.int32), valid

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        case_idx, jaw_flag, block_idx = self.index[idx]
        jaw_key = "upper" if jaw_flag == 0 else "lower"

        blocks6, ids, _valid = self.cache[case_idx][jaw_key]  # [16,1024,6]
        tooth_id = int(ids[block_idx])

        # GT：被删掉那颗牙（1024点，6维）
        ref_label = blocks6[block_idx].copy()

        # 输入：整颌（删掉该牙 -> xyz/normals 都置0）
        ref_blocks = blocks6.copy()
        ref_blocks[block_idx] = 0.0
        ref = ref_blocks.reshape(self.T * self.P, 6)  # [16384,6]

        ref_t = torch.from_numpy(ref).float()
        label_t = torch.from_numpy(ref_label).float()
        meta = torch.tensor([case_idx, jaw_flag, block_idx, tooth_id], dtype=torch.long)

        return ref_t, label_t, meta
