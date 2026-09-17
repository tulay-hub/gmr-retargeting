"""lens110 GMR pkl -> 训练 npy / 部署 CSV 转换脚本。

把 gmr_lbot 工作流 (BVH->G1->lens110) 生成的 pkl 转成:

1. lens110_motion.npy  (N, 21)  float32, 训练用, 默认保持 120 Hz 全部帧
2. lens110_motion_<fps>hz.csv  28 列部署 CSV:
   [root_pos(3) | root_rot xyzw(4) | 21 关节], 每行末尾带逗号, 与旧版一致。
   注意: GMR pkl 里 root_rot 已是 xyzw, 直接透传, 不要转成 wxyz。

用法 (gmr conda 环境):
    python lens110_pkl_to_training.py --pkl /tmp/lens110_test/pkl/tangbohushuoDJ.pkl \
        --out_dir /tmp/lens110_test/training --csv_fps 50
"""

import argparse
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R


# 训练关节顺序 (= lens110 21dof MJCF/URDF 关节顺序)
TRAIN_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def reorder_dof(dof_pos, src_names, dst_order):
    """把 dof_pos (N, K) 从 src_names 顺序重排到 dst_order。"""
    if src_names is None:
        return dof_pos
    src_list = list(src_names)
    if src_list == dst_order:
        return dof_pos
    if set(src_list) != set(dst_order):
        missing = set(dst_order) - set(src_list)
        raise ValueError(f"pkl 关节与训练关节集合不一致, 缺少: {missing}")
    indices = [src_list.index(name) for name in dst_order]
    return dof_pos[:, indices]


def resample_poses(root_pos, root_rot_xyzw, dof_pos, src_fps, dst_fps):
    """把 (N, ...) 的 root_pos/root_rot(xyzw)/dof_pos 重采样到 dst_fps。

    root_rot 用 slerp, 位置/关节用线性插值。
    """
    if dst_fps is None or dst_fps <= 0 or abs(dst_fps - src_fps) < 1e-6:
        return root_pos, root_rot_xyzw, dof_pos
    n = len(root_pos)
    # 目标采样时刻 (以帧为单位, 覆盖 [0, n-1])
    t_out = np.arange(0, n, src_fps / dst_fps)
    t_out = np.clip(t_out, 0, n - 1)

    lo = np.floor(t_out).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (t_out - lo)[:, None]

    new_root_pos = (1 - w) * root_pos[lo] + w * root_pos[hi]
    new_dof_pos = (1 - w) * dof_pos[lo] + w * dof_pos[hi]

    # slerp: xyzw -> scipy (wxyz 标量在前) 处理
    quat_lo = R.from_quat(root_rot_xyzw[lo])
    quat_hi = R.from_quat(root_rot_xyzw[hi])
    omega = np.arccos(np.clip(np.sum(quat_lo.as_quat() * quat_hi.as_quat(), axis=1), -1, 1))
    mask = omega > 1e-6
    result = np.empty((len(t_out), 4))
    result[~mask] = quat_lo.as_quat()[~mask]
    ww = np.zeros(len(t_out))
    ww[mask] = np.sin((1 - w[mask, 0]) * omega[mask]) / np.sin(omega[mask])
    ww2 = np.zeros(len(t_out))
    ww2[mask] = np.sin(w[mask, 0] * omega[mask]) / np.sin(omega[mask])
    result[mask] = (
        ww[mask][:, None] * quat_lo.as_quat()[mask]
        + ww2[mask][:, None] * quat_hi.as_quat()[mask]
    )
    # 归一化并转回 xyzw (scipy as_quat 是 xyzw)
    result = result / np.linalg.norm(result, axis=1, keepdims=True)
    return new_root_pos, result, new_dof_pos


def write_csv_28(root_pos, root_rot_xyzw, dof_pos, path):
    """写 28 列 CSV: root_pos(3) + root_rot xyzw(4) + 21 关节, 行末带逗号。"""
    n = len(root_pos)
    cols = np.concatenate(
        [root_pos.reshape(n, 3), root_rot_xyzw.reshape(n, 4), dof_pos.reshape(n, 21)],
        axis=1,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in cols:
            f.write(",".join(f"{v:.17g}" for v in row) + ",\n")
    print(f"[csv] saved {path}  shape={cols.shape}")


def report_joint_limits(dof_pos, names, path=None):
    """用 MJCF 限位检查每列超限帧数。path 为 None 时跳过。"""
    if path is None:
        return
    try:
        import mujoco as mj
    except ImportError:
        return
    model = mj.MjModel.from_xml_path(path)
    limits = {}
    for i in range(model.njnt):
        jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        if jname in names and model.jnt_limited[i]:
            limits[jname] = model.jnt_range[i]
    print("[limits] 关节限位超限帧数 (期望全为 0):")
    any_over = False
    for idx, name in enumerate(names):
        if name not in limits:
            continue
        lo, hi = limits[name]
        over = int(((dof_pos[:, idx] < lo) | (dof_pos[:, idx] > hi)).sum())
        if over:
            any_over = True
        print(f"  {name:28s} [{lo:+.3f}, {hi:+.3f}]  over={over}")
    if not any_over:
        print("  (全部在限位内)")


def main():
    parser = argparse.ArgumentParser(description="lens110 GMR pkl -> npy + CSV")
    parser.add_argument("--pkl", required=True, help="batch_bvh_to_lens110.py 生成的 pkl")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument(
        "--npy_path",
        default=None,
        help="npy 输出路径 (默认 out_dir/lens110_motion.npy)",
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        help="CSV 输出路径 (默认 out_dir/lens110_motion_<csv_fps>hz.csv)",
    )
    parser.add_argument(
        "--csv_fps",
        type=float,
        default=50.0,
        help="CSV 重采样帧率, 默认 50 (源为 120 Hz BVH)",
    )
    parser.add_argument(
        "--mjcf",
        default=None,
        help="lens110 MJCF 路径, 用于限位检查 (不传则跳过)",
    )
    args = parser.parse_args()

    data = load_pkl(args.pkl)
    src_fps = float(data["fps"])
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    src_names = data.get("dof_names")

    if src_names is not None:
        print(f"[pkl] dof_names ({len(src_names)}): {list(src_names)}")
    print(f"[pkl] frames={len(root_pos)}  fps={src_fps}  dof shape={dof_pos.shape}")

    dof_pos = reorder_dof(dof_pos, src_names, TRAIN_JOINT_ORDER)

    # 训练 npy: 不重采样, 保持源帧率 (120 Hz, 与 observations.py 的 120 Hz 采样一致)
    os.makedirs(args.out_dir, exist_ok=True)
    npy_path = args.npy_path or os.path.join(args.out_dir, "lens110_motion.npy")
    motion = dof_pos.astype(np.float32)
    np.save(npy_path, motion)
    print(f"[npy] saved {npy_path}  shape={motion.shape}  dtype={motion.dtype}")

    # 部署 CSV: 重采样到 csv_fps (root_rot 保持 xyzw, 与旧版/用户 FK 脚本一致)
    rp, rr, df = resample_poses(root_pos, root_rot_xyzw, dof_pos, src_fps, args.csv_fps)
    csv_path = args.csv_path or os.path.join(
        args.out_dir, f"lens110_motion_{int(round(args.csv_fps))}hz.csv"
    )
    write_csv_28(rp, rr, df, csv_path)

    print(f"[csv] root_z range: [{rp[:, 2].min():.4f}, {rp[:, 2].max():.4f}]")
    print(f"[csv] frames={len(rp)}")

    report_joint_limits(df, TRAIN_JOINT_ORDER, args.mjcf)


if __name__ == "__main__":
    main()
