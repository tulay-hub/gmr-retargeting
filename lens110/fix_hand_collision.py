"""修复 lens110 动作中双手末端网格穿模。

背景: lens110 的手部是空 body (无网格), MuJoCo 里可见的"手末端"其实是
left/right_elbow_link 的网格末端。舞蹈里双手在胸前合拢的帧, 两个网格会
相触甚至穿模。

修复方式: 在碰撞窗口内给 left_shoulder_roll 加正偏移、right_shoulder_roll
加负偏移, 把两只手向外分开; 偏移量按穿模程度自适应, 并在窗口边界平滑过渡。
窗口外的动作完全不变。

用法:
    python tools/retargeting/gmr_lens110/lens110/fix_hand_collision.py \
        --pkl projects/01_dance_whole_body/data/processed/retargeted_actions/tangbohushuo_dance/pkl/tangbohushuoDJ_v2.pkl \
        --out-pkl /tmp/tangbohushuoDJ_nohit.pkl \
        --out-csv /tmp/tangbohushuoDJ_nohit.csv
"""

import argparse
import os
import pickle

import mujoco as mj
import numpy as np
from scipy.spatial import cKDTree

from path_utils import lens110_mjcf


LSR_INDEX = 14  # left_shoulder_roll
RSR_INDEX = 18  # right_shoulder_roll


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def mesh_vertices(model, geom_id):
    mid = model.geom_dataid[geom_id]
    a = model.mesh_vertadr[mid]
    n = model.mesh_vertnum[mid]
    return model.mesh_vert[a : a + n].reshape(-1, 3)


def main():
    parser = argparse.ArgumentParser(description="修复双手末端网格穿模")
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out-pkl", required=True)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--mjcf", default=None)
    parser.add_argument("--target-gap", type=float, default=0.05, help="期望手末端间距 (m)")
    parser.add_argument("--max-delta", type=float, default=0.25, help="肩 roll 最大偏移 (rad)")
    parser.add_argument("--smooth", type=int, default=9, help="偏移时间平滑窗口")
    args = parser.parse_args()

    mjcf = args.mjcf or str(lens110_mjcf())
    model = mj.MjModel.from_xml_path(mjcf)
    data = mj.MjData(model)
    gL, gR = 22, 26  # left/right_elbow_link geom
    vertsL = mesh_vertices(model, gL)
    vertsR = mesh_vertices(model, gR)

    d = load_pkl(args.pkl)
    root_pos = np.asarray(d["root_pos"], dtype=np.float64)
    root_rot = np.asarray(d["root_rot"], dtype=np.float64)
    dof = np.asarray(d["dof_pos"], dtype=np.float64).copy()
    n = len(root_pos)

    def mesh_dist_at(i, dof_arr):
        q = np.zeros(28)
        q[:3] = root_pos[i]
        q[3:7] = root_rot[i][[3, 0, 1, 2]]
        q[7:] = dof_arr[i]
        data.qpos[:] = q
        mj.mj_forward(model, data)
        wL = vertsL @ data.geom_xmat[gL].reshape(3, 3).T + data.geom_xpos[gL]
        wR = vertsR @ data.geom_xmat[gR].reshape(3, 3).T + data.geom_xpos[gR]
        return cKDTree(wL).query(wR)[0].min()

    # 1) 全序列粗略中心距离 -> 候选帧
    centers = np.zeros((n, 3))
    for i in range(n):
        q = np.zeros(28)
        q[:3] = root_pos[i]
        q[3:7] = root_rot[i][[3, 0, 1, 2]]
        q[7:] = dof[i]
        data.qpos[:] = q
        mj.mj_forward(model, data)
        centers[i] = data.geom_xpos[gL] - data.geom_xpos[gR]
    cand = np.where(np.linalg.norm(centers, axis=1) < 0.45)[0]
    print(f"[scan] 中心距<0.45m 候选帧 {len(cand)}")

    # 2) 候选帧精确网格距离
    dist = np.full(n, 1.0)
    for i in cand:
        dist[i] = mesh_dist_at(i, dof)
    print(f"[scan] 网格最小距离 {dist.min():.4f} m @帧 {dist.argmin()}")

    # 3) 自适应偏移 (多轮: 拉不够再加大)
    delta = np.zeros(n)
    for _ in range(4):
        need = dist - args.target_gap
        fix = np.where(need < 0)[0]
        if not len(fix):
            break
        add = np.clip(-need * 3.0, 0.0, args.max_delta)
        delta[fix] = np.minimum(delta[fix] + add[fix], args.max_delta)
        # 时间平滑 + 边界渐变
        if args.smooth > 1:
            w = args.smooth
            kernel = np.ones(w) / w
            delta = np.convolve(np.pad(delta, (w // 2, w - 1 - w // 2), mode="edge"),
                                kernel, mode="valid")
        dd = dof.copy()
        dd[:, LSR_INDEX] += delta
        dd[:, RSR_INDEX] -= delta
        for i in cand:
            dist[i] = mesh_dist_at(i, dd)
        print(f"[fix] 轮次: 修改帧数 {len(np.where(delta > 1e-4)[0])}, "
              f"当前最小网格距离 {dist.min():.4f} m")

    # 4) 应用
    dof[:, LSR_INDEX] += delta
    dof[:, RSR_INDEX] -= delta
    n_fixed = int((delta > 1e-4).sum())
    print(f"[done] 修改肩 roll 帧数 {n_fixed} ({n_fixed / d['fps']:.2f}s), "
          f"最大偏移 {delta.max():.3f} rad")

    # 5) 保存
    os.makedirs(os.path.dirname(args.out_pkl) or ".", exist_ok=True)
    motion_data = dict(d)
    motion_data["dof_pos"] = dof
    motion_data["hand_collision_fix"] = {
        "target_gap": args.target_gap,
        "max_delta": args.max_delta,
        "frames_fixed": int(n_fixed),
        "max_delta_applied": float(delta.max()),
        "min_mesh_distance_after": float(dist.min()),
    }
    with open(args.out_pkl, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"[pkl] saved {args.out_pkl}")

    if args.out_csv:
        cols = np.concatenate([root_pos, root_rot, dof], axis=1)
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w") as f:
            for row in cols:
                f.write(",".join(f"{v:.10g}" for v in row) + ",\n")
        print(f"[csv] saved {args.out_csv}")


if __name__ == "__main__":
    main()
