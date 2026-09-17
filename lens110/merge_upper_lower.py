"""合并两段 lens110 动作: 上半身(含腰部) 来自 pkl, 下半身腿部 来自 CSV。

以 pkl 原生时间轴输出 (4143 帧 @120Hz), 播放速度与原生 pkl 完全一致:
    - 时间轴 / 帧数 / fps      来自 --upper-pkl (原生, 不做重采样)
    - root + 12 个腿关节       来自 --lower-csv, 按"整段时长"映射到 pkl 时间轴
    - torso_yaw + 8 个手臂关节 来自 --upper-pkl (原生)
    - --keep-lift: CSV 腿部里被压平的抬腿段 (脚踝离地 > 阈值) 自动用 pkl 腿部
      替换并在边界做渐变过渡, 避免丢掉源动作里的抬腿

输出:
    --out-csv  28 列部署 CSV (与 pkl 同帧数/帧率)
    --out-pkl  GMR pkl (可播放, fps 与 pkl 一致)

用法:
    python merge_upper_lower.py \
        --upper-pkl /tmp/lens110_full/pkl/tangbohushuoDJ.pkl \
        --lower-csv projects/02_dance_half_body/data/processed/retargeted_actions/mixed_k1_mj_motion/k1_mj2_seg1_50fps_lens110_hybrid_50hz.csv \
        --out-csv projects/02_dance_half_body/data/processed/retargeted_actions/mixed_k1_mj_motion/merged_120hz.csv \
        --out-pkl projects/02_dance_half_body/data/processed/retargeted_actions/mixed_k1_mj_motion/merged_120hz.pkl
"""

import argparse
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R

from path_utils import lens110_mjcf


LEG_SLICE = slice(0, 12)      # dof 前 12 列 = 腿关节
UPPER_SLICE = slice(12, 21)   # dof 后 9 列 = torso + 手臂


def load_csv(path):
    with open(path) as f:
        rows = [[float(x) for x in line.rstrip().rstrip(",").split(",")] for line in f if line.strip()]
    arr = np.array(rows)
    if arr.shape[1] != 28:
        raise RuntimeError(f"期望 28 列, 实际 {arr.shape[1]}")
    return arr


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def interp_columns(src, n_out):
    """把 (n_in, k) 按首尾时长线性插值到 (n_out, k)。"""
    t = np.arange(n_out) * (len(src) - 1) / max(n_out - 1, 1)
    return np.column_stack([np.interp(t, np.arange(len(src)), src[:, j]) for j in range(src.shape[1])])


def slerp_quat_xyzw(src, n_out):
    """把 (n_in, 4) xyzw 四元数序列 slerp 到 n_out 帧, 返回 xyzw。"""
    t = np.arange(n_out) * (len(src) - 1) / max(n_out - 1, 1)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(src) - 1)
    w = (t - lo)[:, None]
    q_lo = src[lo]
    q_hi = src[hi]
    # 处理符号歧义: 取点积为正的方向
    dots = np.sum(q_lo * q_hi, axis=1, keepdims=True)
    q_hi = np.where(dots < 0, -q_hi, q_hi)
    omega = np.arccos(np.clip(np.sum(q_lo * q_hi, axis=1), -1, 1))
    mask = omega > 1e-6
    result = np.empty((n_out, 4))
    result[~mask] = q_lo[~mask]
    a = np.zeros(n_out)
    b = np.zeros(n_out)
    a[mask] = np.sin((1 - w[mask, 0]) * omega[mask]) / np.sin(omega[mask])
    b[mask] = np.sin(w[mask, 0] * omega[mask]) / np.sin(omega[mask])
    result[mask] = a[mask][:, None] * q_lo[mask] + b[mask][:, None] * q_hi[mask]
    result = result / np.linalg.norm(result, axis=1, keepdims=True)
    return result


def find_lift_regions(model, root_pos, root_rot_wxyz, dof_pos, threshold=0.08, margin=24):
    """用 FK 找脚踝离地 > threshold 的连续段 (pkl 帧索引), 加 margin 膨胀。"""
    import mujoco as mj

    data = mj.MjData(model)
    left_id = model.body("left_ankle_roll_link").id
    right_id = model.body("right_ankle_roll_link").id
    n = len(root_pos)
    mask = np.zeros(n, dtype=bool)
    q = np.zeros(model.nq)
    for i in range(n):
        q[:3] = root_pos[i]
        q[3:7] = root_rot_wxyz[i]
        q[7:] = dof_pos[i]
        data.qpos[:] = q
        mj.mj_forward(model, data)
        mask[i] = max(data.xpos[left_id][2], data.xpos[right_id][2]) > threshold
    # 膨胀
    for _ in range(margin):
        mask[:-1] |= mask[1:]
        mask[1:] |= mask[:-1].copy()
    return mask


def main():
    parser = argparse.ArgumentParser(description="合并 pkl 上半身 + CSV 下半身 (pkl 原生时间轴)")
    parser.add_argument("--upper-pkl", required=True)
    parser.add_argument("--lower-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-pkl", required=True)
    parser.add_argument("--mjcf", default=None, help="lens110 MJCF (抬腿检测用)")
    parser.add_argument("--no-keep-lift", dest="keep_lift", action="store_false", default=True,
                        help="关闭抬腿保留: 腿部全程使用 CSV")
    parser.add_argument("--lift-threshold", type=float, default=0.08, help="抬腿检测阈值 (m)")
    args = parser.parse_args()

    pkl = load_pkl(args.upper_pkl)
    csv = load_csv(args.lower_csv)
    n = len(pkl["root_pos"])
    pkl_fps = float(pkl.get("fps", 120.0))

    pkl_root_pos = np.asarray(pkl["root_pos"], dtype=np.float64)
    pkl_root_rot = np.asarray(pkl["root_rot"], dtype=np.float64)  # xyzw
    pkl_dof = np.asarray(pkl["dof_pos"], dtype=np.float64)

    # CSV -> pkl 时间轴: 整段时长映射 (首尾对齐, 覆盖全程)
    csv_legs = interp_columns(csv[:, 7:19], n)
    csv_root_pos = interp_columns(csv[:, :3], n)
    csv_root_rot = slerp_quat_xyzw(csv[:, 3:7], n)
    print(f"[align] pkl {n}f@{pkl_fps:.0f}Hz, csv {len(csv)}f -> 映射到 pkl 时间轴 (整段时长覆盖)")

    # 合并: root + 腿来自 CSV, 上半身来自 pkl
    merged_dof = np.empty_like(pkl_dof)
    merged_dof[:, LEG_SLICE] = csv_legs
    merged_dof[:, UPPER_SLICE] = pkl_dof[:, UPPER_SLICE]

    if args.keep_lift:
        import mujoco as mj

        mjcf = args.mjcf or str(lens110_mjcf())
        model = mj.MjModel.from_xml_path(mjcf)
        lift_mask = find_lift_regions(
            model, pkl_root_pos, pkl_root_rot[:, [3, 0, 1, 2]], pkl_dof,
            threshold=args.lift_threshold,
        )
        if lift_mask.any():
            # 边界渐变 (每侧 blend 24 帧)
            blend = 24
            lift_idx = np.where(lift_mask)[0]
            start, end = lift_idx[0], lift_idx[-1]
            s0, e0 = max(0, start - blend), min(n, end + blend)
            for j in range(12):
                for i in range(start, end + 1):
                    merged_dof[i, j] = pkl_dof[i, j]
                # 左过渡
                for k, i in enumerate(range(s0, start)):
                    w = (k + 1) / (start - s0 + 1)
                    merged_dof[i, j] = (1 - w) * csv_legs[i, j] + w * pkl_dof[i, j]
                # 右过渡
                for k, i in enumerate(range(end + 1, e0)):
                    w = 1 - (k + 1) / (e0 - end)
                    merged_dof[i, j] = (1 - w) * csv_legs[i, j] + w * pkl_dof[i, j]
            print(f"[lift] 抬腿段 pkl 帧 {start}-{end} ({(end - start + 1) / pkl_fps:.2f}s), "
                  f"已用 pkl 腿部替换并做 {blend} 帧过渡")
        else:
            print("[lift] 未检测到抬腿段 (阈值 %.2fm), 全部使用 CSV 腿部" % args.lift_threshold)

    merged_root_pos = csv_root_pos
    merged_root_rot = csv_root_rot  # xyzw

    # 写 CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    cols = np.concatenate([merged_root_pos, merged_root_rot, merged_dof], axis=1)
    with open(args.out_csv, "w") as f:
        for row in cols:
            f.write(",".join(f"{v:.10g}" for v in row) + ",\n")
    print(f"[csv] saved {args.out_csv}  shape={cols.shape}")

    # 写 pkl
    os.makedirs(os.path.dirname(args.out_pkl) or ".", exist_ok=True)
    motion_data = {
        "fps": pkl_fps,
        "root_pos": merged_root_pos,
        "root_rot": merged_root_rot,  # xyzw
        "dof_pos": merged_dof,
        "local_body_pos": None,
        "link_body_list": None,
        "source_upper": str(args.upper_pkl),
        "source_lower": str(args.lower_csv),
    }
    with open(args.out_pkl, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"[pkl] saved {args.out_pkl}  frames={n} fps={pkl_fps}")

    print(f"[check] upper range: [{merged_dof[:, UPPER_SLICE].min():.3f}, {merged_dof[:, UPPER_SLICE].max():.3f}]")
    print(f"[check] legs  range: [{merged_dof[:, LEG_SLICE].min():.3f}, {merged_dof[:, LEG_SLICE].max():.3f}]")
    dq = np.abs(np.diff(merged_dof, axis=0)).max(axis=0)
    print(f"[check] 最大帧间关节变化: {dq.max():.4f} rad/frame")


if __name__ == "__main__":
    main()
