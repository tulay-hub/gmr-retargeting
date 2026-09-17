"""一键入口: BVH 舞蹈动作 -> lens110 重定向 -> 训练 npy + 部署 CSV。

流程与 gmr_lbot 相同 (对方在用同一款 lens110 机器人且已验证):
    BVH(nokov) -> unitree_g1 (GMR IK) -> lens110_21dof (hybrid 映射)
    -> 贴地/站距/后处理 -> pkl -> lens110_motion.npy + 50 Hz CSV

用法 (在仓库根目录, gmr conda 环境):
    conda activate gmr
    python tools/retargeting/gmr_lens110/lens110/convert_bvh_to_lens110.py \
        --bvh datasets/dance/raw_bvh/original_bvh/tangbohushuoDJ.bvh \
        --out_dir /tmp/lens110_convert

常用参数:
    --retarget_fps 120      # 保持 120 Hz 全部帧 (训练 npy 用)
    --csv_fps 50            # 部署 CSV 输出帧率
    --max_source_frames 300 # 快速试跑时限制源帧数
    --install_motion        # 把 npy 复制到训练 motion 目录
"""

import argparse
import pathlib
import shutil

import sys

HERE = pathlib.Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from batch_bvh_to_lens110 import build_lens110_motion  # noqa: E402
from lens110_pkl_to_training import (  # noqa: E402
    TRAIN_JOINT_ORDER,
    load_pkl,
    reorder_dof,
    report_joint_limits,
    resample_poses,
    write_csv_28,
)
from path_utils import lens110_mjcf, project_training_dir  # noqa: E402


def default_target_xml():
    """Use the canonical MJCF shared by the reorganized training workspace."""
    return str(lens110_mjcf())


def main():
    parser = argparse.ArgumentParser(description="BVH -> lens110 -> npy + CSV")
    parser.add_argument("--bvh", required=True, help="源 BVH 文件")
    parser.add_argument("--out_dir", required=True, help="输出目录 (pkl/txt/training)")
    parser.add_argument("--target_xml", default=None, help="Lens110 MJCF, 默认使用共享训练基座的 canonical MJCF")
    parser.add_argument("--bvh_format", choices=["lafan1", "nokov"], default="nokov")
    parser.add_argument("--retarget_fps", type=float, default=120.0, help="重定向输出帧率 (120=不降采样)")
    parser.add_argument("--csv_fps", type=float, default=50.0)
    parser.add_argument("--method", choices=["direct", "hybrid"], default="hybrid")
    parser.add_argument("--hybrid_iterations", type=int, default=4)
    parser.add_argument("--stance_width_offset", type=float, default=0.10)
    parser.add_argument("--hip_yaw_out_offset", type=float, default=0.03)
    parser.add_argument("--min_stance_width", type=float, default=0.20)
    parser.add_argument("--ground_mode", choices=["constant", "per_frame", "smooth", "off"], default="per_frame")
    parser.add_argument("--ground_clearance", type=float, default=0.0)
    parser.add_argument("--max_ground_lowering", type=float, default=0.20)
    parser.add_argument("--root_xy_scale", type=float, default=0.72)
    parser.add_argument("--skip_initial_frames", type=int, default=1)
    parser.add_argument("--max_source_frames", type=int, default=None)
    parser.add_argument("--root_z_offset", type=float, default=0.0)
    parser.add_argument("--post_lower_floating_feet", action="store_true", default=False)
    parser.add_argument("--arm_smooth_window", type=int, default=3)
    parser.add_argument(
        "--max_arm_joint_step", type=float, default=None,
        help="手臂关节每帧最大角度变化 (rad)。0.087=120Hz 下 10.4 rad/s 限速; "
             "None 表示不限速",
    )
    parser.add_argument("--install_motion", action="store_true", help="把 npy 安装到训练 motion 目录")
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    bvh = pathlib.Path(args.bvh)
    out_dir = pathlib.Path(args.out_dir)
    target_xml = args.target_xml or default_target_xml()

    pkl_path = out_dir / "pkl" / f"{bvh.stem}.pkl"
    txt_path = out_dir / "txt" / f"{bvh.stem}.txt"

    print(f"[1/2] BVH -> lens110 重定向 (target_xml={target_xml})")
    stats = build_lens110_motion(
        bvh_path=bvh,
        pkl_path=pkl_path,
        txt_path=txt_path,
        target_xml=target_xml,
        bvh_format=args.bvh_format,
        target_fps=args.retarget_fps,
        max_source_frames=args.max_source_frames,
        skip_initial_frames=args.skip_initial_frames,
        method=args.method,
        hybrid_iterations=args.hybrid_iterations,
        stance_width_offset=args.stance_width_offset,
        hip_yaw_out_offset=args.hip_yaw_out_offset,
        leg_motion_scale=1.0,
        hip_yaw_scale=1.0,
        ankle_roll_scale=1.0,
        torso_yaw_scale=1.0,
        lower_body_smooth_window=3,
        min_stance_width=args.min_stance_width,
        stance_correction_gain=3.5,
        stance_correction_iterations=6,
        max_stance_correction_step=0.16,
        ground_mode=args.ground_mode,
        ground_clearance=args.ground_clearance,
        foot_ground_offset=0.0,
        ground_percentile=2.0,
        max_ground_lowering=args.max_ground_lowering,
        ground_smooth_window=7,
        root_z_offset=args.root_z_offset,
        root_xy_scale=args.root_xy_scale,
        root_roll_scale=1.0,
        root_pitch_scale=1.0,
        max_lateral_roll=0.0,
        max_forward_pitch=0.0,
        max_backward_pitch=0.0,
        root_rot_smooth_window=1,
        root_z_smooth_window=1,
        root_z_motion_scale=1.0,
        post_lower_floating_feet=args.post_lower_floating_feet,
        max_post_ground_lowering=None,
        arm_smooth_window=args.arm_smooth_window,
        max_arm_joint_step=args.max_arm_joint_step,
        verbose=args.verbose,
    )
    print(f"      frames={stats['frames']} fps={stats['fps']:.1f}")

    print("[2/2] pkl -> 训练 npy + 部署 CSV")
    data = load_pkl(pkl_path)
    src_fps = float(data["fps"])
    root_pos = data["root_pos"]
    root_rot_xyzw = data["root_rot"]
    dof_pos = reorder_dof(data["dof_pos"], data.get("dof_names"), TRAIN_JOINT_ORDER)

    training_dir = out_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    npy_path = training_dir / "lens110_motion.npy"
    import numpy as np

    np.save(npy_path, dof_pos.astype(np.float32))
    print(f"[npy] {npy_path}  shape={dof_pos.shape}  fps={src_fps}")

    rp, rr, df = resample_poses(root_pos, root_rot_xyzw, dof_pos, src_fps, args.csv_fps)
    csv_path = training_dir / f"lens110_motion_{int(round(args.csv_fps))}hz.csv"
    write_csv_28(rp, rr, df, str(csv_path))
    print(f"[csv] root_z range: [{rp[:, 2].min():.4f}, {rp[:, 2].max():.4f}]  frames={len(rp)}")
    report_joint_limits(df, TRAIN_JOINT_ORDER, target_xml)

    if args.install_motion:
        train_motion = project_training_dir() / "lens110_motion.npy"
        train_motion.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(npy_path, train_motion)
        print(f"[install] copied npy -> {train_motion}")

    print("完成。产物: pkl/txt (GMR 回放), training/lens110_motion.npy (训练), "
          f"training/lens110_motion_{int(round(args.csv_fps))}hz.csv (部署)")


if __name__ == "__main__":
    main()
