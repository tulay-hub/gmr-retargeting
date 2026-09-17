"""无 viewer 验证脚本: BVH → lens110 pkl (快速, 无头, 不开 mujoco 窗口)。

用于快速验证 GMR 重定向配置是否正确, 跳过 RobotMotionViewer。
生成 pkl 后, 使用 `tools/retargeting/gmr_lens110/lens110/lens110_pkl_to_training.py` 转为训练 npy。

注意: BVH 经过 load_bvh_file 转换后坐标系为 (X左右, Y后, Z上),
而 lens110 机器人坐标系为 (X前, Y左, Z上)。
需要在加载后绕 Z 轴 +90 度旋转位置和朝向, 使两者对齐。
"""
import argparse
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_bvh_file


# BVH(lafan1.py处理后) → lens110 坐标系转换
# lafan1.py转换后: BVH = (X右, Y后, Z上) — 右手系
# 机器人: (X前, Y左, Z上) — 左手系
# 手性不同，需要: 1) 位置翻转Y轴 (X右,Y后,Z上 → X右,Y前,Z上)
#               2) 绕Z轴-90度旋转 (X右,Y前,Z上 → X前,Y左,Z上)
# 朝向用旋转处理（四元数无法表示反射）
COORD_ROT_QUAT = R.from_euler("z", -90, degrees=True).as_quat(scalar_first=True)


def transform_bvh_coord(data_frames):
    """将 BVH 数据从 (X右, Y后, Z上) 转到 (X前, Y左, Z上)。

    位置: 先翻转Y轴，再旋转
    朝向: 只用旋转（R*q*R^(-1)）
    """
    rot = R.from_quat(COORD_ROT_QUAT, scalar_first=True)
    rot_inv = rot.inv()
    for frame in data_frames:
        for body_name in frame:
            pos = frame[body_name][0].copy()
            quat = frame[body_name][1]  # wxyz
            # 位置: 先翻转Y轴
            pos[1] = -pos[1]
            # 再旋转
            new_pos = rot.apply(pos)
            # 朝向: 只用旋转
            new_quat = (rot * R.from_quat(quat, scalar_first=True) * rot_inv).as_quat(scalar_first=True)
            frame[body_name] = [new_pos, new_quat]
    return data_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", required=True)
    parser.add_argument("--robot", default="lens110")
    parser.add_argument("--format", default="lafan1", choices=["lafan1", "nokov"])
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--motion_fps", type=int, default=30)
    parser.add_argument("--no_coord_transform", action="store_true",
                        help="跳过坐标系转换 (默认启用, 适配 lens110)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    data_frames, human_height = load_bvh_file(args.bvh_file, format=args.format)
    print(f"[load] {len(data_frames)} frames, human_height={human_height:.3f}")

    # 坐标系转换: BVH (X左右,Y后,Z上) → robot (X前,Y左,Z上)
    if not args.no_coord_transform:
        data_frames = transform_bvh_coord(data_frames)
        print("[coord] 已将 BVH 坐标系转到机器人坐标系 (绕Z轴+90度，只旋转位置)")

    retargeter = GMR(
        src_human=f"bvh_{args.format}",
        tgt_robot=args.robot,
        actual_human_height=human_height,
    )

    qpos_list = []
    for i in tqdm(range(len(data_frames)), desc="retarget"):
        qpos = retargeter.retarget(data_frames[i])
        qpos_list.append(qpos)

    qpos = np.array(qpos_list)
    motion_data = {
        "fps": args.motion_fps,
        "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7],
        "dof_pos": qpos[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }
    with open(args.save_path, "wb") as f:
        pickle.dump(motion_data, f)
    dof = motion_data["dof_pos"]
    print(f"[saved] {args.save_path}")
    print(f"  dof_pos shape: {dof.shape}")
    print(f"  dof_pos range: [{dof.min():.3f}, {dof.max():.3f}]")


if __name__ == "__main__":
    main()
