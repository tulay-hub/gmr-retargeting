"""修复 lens110 部署 CSV 左右肩 roll/yaw 列互换的问题。

现象: 播放时左右胳膊反了。原因是 left_shoulder_roll / left_shoulder_yaw
列里实际装的是右臂数据, right_* 列装的是左臂数据 (肩 pitch 和肘不受影响)。

修复: 交换 lsr<->rsr, lsy<->rsy 两对列, 其余列原样保留。

用法:
    python tools/retargeting/gmr_lens110/lens110/fix_lens110_arm_swap.py \
        --csv projects/01_dance_whole_body/data/processed/retargeted_actions/tangbohushuo_dance/training/tangbohushuoDJ_v2_100hz.csv \
        --out /tmp/tangbohushuoDJ_v2_100hz_fixed.csv
"""

import argparse
import os

import numpy as np


# 28 列: 0-2 root_pos, 3-6 root_rot xyzw, 7-27 关节
# 关节顺序: lhp lhr lhy lk lankp lankr rhp rhr rhy rk rankp rankr
#           torso lsp lsr lsy lelb rsp rsr rsy relb
SWAP_PAIRS = [(21, 25), (22, 26)]  # lsr<->rsr, lsy<->rsy


def load_csv(path):
    with open(path) as f:
        rows = [[float(x) for x in line.rstrip().rstrip(",").split(",")] for line in f if line.strip()]
    arr = np.array(rows)
    if arr.shape[1] != 28:
        raise RuntimeError(f"期望 28 列, 实际 {arr.shape[1]}")
    return arr


def main():
    parser = argparse.ArgumentParser(description="修复左右肩 roll/yaw 列互换")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mjcf", default=None, help="lens110 MJCF, 用于限位校验")
    args = parser.parse_args()

    csv = load_csv(args.csv)
    fixed = csv.copy()
    for a, b in SWAP_PAIRS:
        fixed[:, [a, b]] = fixed[:, [b, a]]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for row in fixed:
            f.write(",".join(f"{v:.10g}" for v in row) + ",\n")
    print(f"[saved] {args.out}  shape={fixed.shape}")

    if args.mjcf:
        import mujoco as mj

        model = mj.MjModel.from_xml_path(args.mjcf)
        names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "torso_yaw_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint", "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        ]
        limits = {}
        for i in range(model.njnt):
            jn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
            if jn in names and model.jnt_limited[i]:
                limits[jn] = model.jnt_range[i]
        print("[limits] 修复后超限帧数:")
        total_over = 0
        for j, nm in enumerate(names):
            if nm not in limits:
                continue
            lo, hi = limits[nm]
            over = int(((fixed[:, 7 + j] < lo) | (fixed[:, 7 + j] > hi)).sum())
            total_over += over
            if over:
                print(f"  {nm:28s} over={over}")
        print(f"  总计超限: {total_over} 帧" if total_over else "  全部在限位内")


if __name__ == "__main__":
    main()
