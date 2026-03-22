import os
import re
import argparse
import numpy as np

try:
    import trimesh
except ImportError as e:
    raise ImportError("请先安装依赖: pip install trimesh") from e


# -----------------------------
# 1) 解析牙位编号（适配：牙齿_11_缺失牙_-_术前模型.stl）
# -----------------------------
def parse_tooth_id(filename: str) -> int:
    """
    支持命名示例：
      牙齿_11_缺失牙_-_术前模型.stl  -> 11
      牙齿_31_邻牙_-_术前模型.stl    -> 31
    也兼容旧格式：
      ...-46-tooth... / tooth_46 ...
    """
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]

    # 1) 最优先：中文前缀“牙齿_11_...”（或牙齿-11-...）
    m = re.search(r"牙齿[_-](\d{2})(?=[_-])", name)
    if m:
        tid = int(m.group(1))
        if 11 <= tid <= 48:
            return tid

    # 2) 兼容旧格式："-46-tooth"
    m = re.search(r"-(\d{2})(?=-tooth)", name)
    if m:
        tid = int(m.group(1))
        if 11 <= tid <= 48:
            return tid

    # 3) 兼容旧格式：tooth_46 / Tooth-46
    m = re.search(r"(?:tooth|Tooth)[_-]?(\d{2})", name)
    if m:
        tid = int(m.group(1))
        if 11 <= tid <= 48:
            return tid

    # 4) 兜底：从所有两位数里挑一个符合FDI(11-48)的
    all_2d = [int(x) for x in re.findall(r"(\d{2})", name)]
    cand = [x for x in all_2d if 11 <= x <= 48]
    if cand:
        # 建议取第一个符合FDI的两位数，避免日期/版本号干扰
        return cand[0]

    raise ValueError(f"无法从文件名提取牙位号(11-48): {filename}")


def jaw_from_tooth_id(tooth_id: int) -> str:
    """
    FDI 编号：
    1/2 象限 = 上颌
    3/4 象限 = 下颌
    """
    s = str(tooth_id)
    if len(s) != 2:
        raise ValueError(f"牙位号格式不对: {tooth_id}")
    first_digit = int(s[0])
    if first_digit in (1, 2):
        return "upper"
    if first_digit in (3, 4):
        return "lower"
    raise ValueError(f"牙位号不符合 FDI(11-48): {tooth_id}")


# -----------------------------
# 2) STL Mesh -> 点云（面上采样）
# -----------------------------
def load_mesh(stl_path: str):
    mesh = trimesh.load(stl_path, force="mesh")

    # 有些 stl 会读成 Scene
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"空 Scene: {stl_path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"无法解析为 Trimesh: {stl_path}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"无有效顶点: {stl_path}")

    # 轻量清理（不强求 watertight）
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()

    return mesh


def sample_surface_points(mesh, n_samples: int) -> np.ndarray:
    """
    在 mesh 表面按面积采样点
    """
    pts = mesh.sample(n_samples)
    return np.asarray(pts, dtype=np.float32)


# -----------------------------
# 3) FPS 最远点采样（numpy版）
# -----------------------------
def farthest_point_sampling(points: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    points: [N,3]
    return: [k,3]
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points 维度必须是 [N,3]，实际: {points.shape}")

    if points.shape[0] < k:
        raise ValueError(f"点数不足以做 FPS: N={points.shape[0]} < k={k}")

    # 初始点：离质心最远点（更稳定）
    centroid = points.mean(axis=0, keepdims=True)
    dist2 = np.sum((points - centroid) ** 2, axis=1)
    farthest = int(np.argmax(dist2))

    selected_idx = np.empty((k,), dtype=np.int64)
    selected_idx[0] = farthest

    # 每个点到已选集合的最小距离
    min_dist2 = np.sum((points - points[farthest]) ** 2, axis=1)

    for i in range(1, k):
        farthest = int(np.argmax(min_dist2))
        selected_idx[i] = farthest

        d2 = np.sum((points - points[farthest]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, d2)

    return points[selected_idx]


# -----------------------------
# 4) 牙位排序（默认输出 16 颗/颌：含智齿）
# -----------------------------
def get_expected_order(jaw: str, include_wisdom: bool) -> list[int]:
    """
    返回该颌拼接顺序（稳定可复现）。
    include_wisdom=True：输出16颗/颌
    include_wisdom=False：输出14颗/颌（去智齿）
    """
    if jaw == "upper":
        full = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
        return full if include_wisdom else [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27]

    if jaw == "lower":
        full = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
        return full if include_wisdom else [47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37]

    raise ValueError("jaw 必须是 upper 或 lower")


# -----------------------------
# 5) 主流程：每牙1024点 -> 拼接 -> 保存
# -----------------------------
def process_folder_stl(
    input_dir: str,
    output_dir: str,
    fps_k: int = 1024,
    surface_samples: int = 20000,
    include_wisdom: bool = True,
    seed: int = 0,
    recursive: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    stl_files = []
    if recursive:
        for root, _dirs, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(".stl"):
                    stl_files.append(os.path.join(root, f))
    else:
        stl_files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(".stl")
        ]

    if not stl_files:
        raise FileNotFoundError(f"未找到 .stl 文件: {input_dir}")

    # jaw -> tooth_id -> path
    jaw_map = {"upper": {}, "lower": {}}
    for p in stl_files:
        tid = parse_tooth_id(p)
        jaw = jaw_from_tooth_id(tid)
        if tid in jaw_map[jaw]:
            print(f"[WARN] 同一牙位重复文件：{tid}，将用最后一个覆盖：{p}")
        jaw_map[jaw][tid] = p

    for jaw in ("upper", "lower"):
        expected_order = get_expected_order(jaw, include_wisdom=include_wisdom)

        all_blocks = []
        labels = []
        missing = []

        for tid in expected_order:
            if tid not in jaw_map[jaw]:
                missing.append(tid)
                all_blocks.append(np.zeros((fps_k, 3), dtype=np.float32))
                labels.append(tid)
                continue

            stl_path = jaw_map[jaw][tid]
            mesh = load_mesh(stl_path)

            pts = sample_surface_points(mesh, surface_samples)
            pts_fps = farthest_point_sampling(pts, fps_k, seed=seed)

            all_blocks.append(pts_fps.astype(np.float32))
            labels.append(tid)

        if missing:
            print(f"[WARN] {jaw} 缺失牙位 {missing}，已用全0点云补齐。")

        teeth = np.concatenate(all_blocks, axis=0)   # [16*1024, 3] 或 [14*1024, 3]
        label_arr = np.array(labels, dtype=np.int32)  # [16] 或 [14]

        # ⚠️ 注意：为了兼容你给的 Teeth3D dataloader，文件名必须固定为下面四个
        teeth_path = os.path.join(output_dir, f"{jaw}_teeth.txt")
        label_path = os.path.join(output_dir, f"{jaw}_label.txt")

        np.savetxt(teeth_path, teeth, fmt="%.6f")
        with open(label_path, "w", encoding="utf-8") as f:
            for t in label_arr.tolist():
                f.write(f"{t}\n")

        print(f"[OK] {jaw}: teeth {teeth.shape} -> {teeth_path}")
        print(f"[OK] {jaw}: label {label_arr.shape} -> {label_path}")


def main():
    parser = argparse.ArgumentParser("STL teeth -> point cloud txt (compatible with Teeth3D dataloader)")

    parser.add_argument(
        "--input_dir",
        type=str,
        default=r"D:\2023\csl\ZGYKD_data\segment2",
        help="包含每颗牙 .stl 的文件夹（文件名形如：牙齿_11_缺失牙_-_术前模型.stl）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\2023\csl\ZGYKD_data\test2",
        help="输出 case 目录（里面会生成 upper/lower 的 txt+label）"
    )

    parser.add_argument("--fps_k", type=int, default=1024, help="每颗牙 FPS 点数")
    parser.add_argument("--surface_samples", type=int, default=20000, help="mesh 表面预采样点数(越大越稳但越慢)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recursive", action="store_true", help="递归扫描 input_dir 下所有 stl")

    parser.add_argument("--include_wisdom", action="store_true", help="输出16颗/颌（含智齿，缺失补0）")
    parser.add_argument("--exclude_wisdom", action="store_true", help="输出14颗/颌（不含智齿）")

    args = parser.parse_args()

    # 默认：含智齿（16颗/颌），与 Teeth3D 的 16384 点一致
    include_wisdom = True
    if args.include_wisdom and args.exclude_wisdom:
        raise ValueError("不要同时传 --include_wisdom 和 --exclude_wisdom")
    if args.exclude_wisdom:
        include_wisdom = False
    if args.include_wisdom:
        include_wisdom = True

    process_folder_stl(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps_k=args.fps_k,
        surface_samples=args.surface_samples,
        include_wisdom=include_wisdom,
        seed=args.seed,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
