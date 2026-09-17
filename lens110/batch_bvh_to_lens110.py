import argparse
import pathlib
import pickle
import re
import sys

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT

HERE = pathlib.Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from batch_g1_csv_to_lens110 import (  # noqa: E402
    apply_lens110_motion_postprocess,
    apply_root_xy_scale,
    direct_map_g1_to_lens110,
    enforce_min_stance_width,
    ground_qpos_by_feet,
    hybrid_map_g1_to_lens110,
    qpos_dof_names,
)
from batch_smplx_to_lens110 import get_foot_mesh_min_z  # noqa: E402
from lens110_gmr_to_legged_lab import convert_lens110_gmr_to_legged_lab  # noqa: E402
from general_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: E402
from path_utils import lens110_mjcf, repository_root  # noqa: E402


def read_bvh_fps(path):
    with open(path, "r") as file:
        for line in file:
            if line.strip().startswith("Frame Time:"):
                frame_time = float(line.split()[2])
                if frame_time > 0.0:
                    return 1.0 / frame_time
    return 30.0


def sanitize_stem(stem):
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "motion"


def sample_frames(frames, source_fps, target_fps, max_source_frames=None):
    if max_source_frames is not None and max_source_frames > 0:
        frames = frames[:max_source_frames]
    if target_fps is None or target_fps <= 0.0 or abs(target_fps - source_fps) < 1e-3:
        return frames, float(source_fps)
    step = max(1, int(round(source_fps / target_fps)))
    return frames[::step], float(source_fps / step)


def retarget_bvh_to_g1_qpos(bvh_path, bvh_format, target_fps, max_source_frames, skip_initial_frames, verbose=False):
    source_fps = read_bvh_fps(bvh_path)
    frames, human_height = load_bvh_file(str(bvh_path), format=bvh_format)
    if skip_initial_frames > 0:
        frames = frames[int(skip_initial_frames) :]
    if not frames:
        raise ValueError(f"{bvh_path} has no frames after skipping {skip_initial_frames} initial frames")
    frames, output_fps = sample_frames(frames, source_fps, target_fps, max_source_frames)
    retargeter = GMR(
        src_human=f"bvh_{bvh_format}",
        tgt_robot="unitree_g1",
        actual_human_height=human_height,
        verbose=verbose,
    )
    qpos_array = np.asarray(
        [
            retargeter.retarget(frame, offset_to_ground=True).copy()
            for frame in tqdm(frames, desc=f"bvh->g1 {bvh_path.name}", leave=False)
        ],
        dtype=np.float64,
    )
    return qpos_array, output_fps


def smooth_1d(values, window):
    window = max(1, int(window))
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def apply_root_upright_postprocess(
    qpos_array,
    root_roll_scale,
    root_pitch_scale,
    max_lateral_roll,
    max_forward_pitch,
    max_backward_pitch,
    root_rot_smooth_window,
):
    if (
        root_roll_scale == 1.0
        and root_pitch_scale == 1.0
        and max_lateral_roll <= 0.0
        and max_forward_pitch <= 0.0
        and max_backward_pitch <= 0.0
        and root_rot_smooth_window <= 1
    ):
        return qpos_array

    processed_qpos = qpos_array.copy()
    root_rot = R.from_quat(processed_qpos[:, 3:7][:, [1, 2, 3, 0]])
    euler = np.unwrap(root_rot.as_euler("xyz"), axis=0)

    if root_rot_smooth_window > 1:
        for axis in range(3):
            euler[:, axis] = smooth_1d(euler[:, axis], root_rot_smooth_window)

    if root_roll_scale != 1.0:
        roll_center = float(np.median(euler[:, 0]))
        euler[:, 0] = roll_center + root_roll_scale * (euler[:, 0] - roll_center)
    if max_lateral_roll > 0.0:
        euler[:, 0] = np.clip(euler[:, 0], -float(max_lateral_roll), float(max_lateral_roll))

    if root_pitch_scale != 1.0:
        pitch_center = float(np.median(euler[:, 1]))
        euler[:, 1] = pitch_center + root_pitch_scale * (euler[:, 1] - pitch_center)

    if max_forward_pitch > 0.0:
        euler[:, 1] = np.maximum(euler[:, 1], -float(max_forward_pitch))
    if max_backward_pitch > 0.0:
        euler[:, 1] = np.minimum(euler[:, 1], float(max_backward_pitch))

    processed_qpos[:, 3:7] = R.from_euler("xyz", euler).as_quat()[:, [3, 0, 1, 2]]
    return processed_qpos


def apply_root_z_postprocess(qpos_array, root_z_smooth_window, root_z_motion_scale):
    if root_z_smooth_window <= 1 and root_z_motion_scale == 1.0:
        return qpos_array
    processed_qpos = qpos_array.copy()
    root_z = processed_qpos[:, 2]
    if root_z_smooth_window > 1:
        root_z = smooth_1d(root_z, root_z_smooth_window)
    root_z_center = float(np.median(root_z))
    processed_qpos[:, 2] = root_z_center + root_z_motion_scale * (root_z - root_z_center)
    return processed_qpos


def lower_floating_feet_only(
    model,
    qpos_array,
    foot_body_names,
    ground_clearance,
    foot_ground_offset,
    max_post_ground_lowering,
):
    if len(qpos_array) == 0:
        return qpos_array
    processed_qpos = qpos_array.copy()
    foot_min_z = get_foot_mesh_min_z(model, processed_qpos, foot_body_names) - foot_ground_offset
    offsets = np.maximum(foot_min_z - ground_clearance, 0.0)
    if max_post_ground_lowering is not None and max_post_ground_lowering > 0.0:
        offsets = np.minimum(offsets, max_post_ground_lowering)
    processed_qpos[:, 2] -= offsets
    return processed_qpos


def clip_joint_limits(model, qpos_array):
    for joint_id in range(model.njnt):
        qpos_addr = model.jnt_qposadr[joint_id]
        if qpos_addr < 7 or not model.jnt_limited[joint_id]:
            continue
        lower, upper = model.jnt_range[joint_id]
        qpos_array[:, qpos_addr] = np.clip(qpos_array[:, qpos_addr], lower, upper)


def apply_arm_postprocess(model, qpos_array, arm_smooth_window, max_arm_joint_step):
    if arm_smooth_window <= 1 and (max_arm_joint_step is None or max_arm_joint_step <= 0.0):
        return qpos_array

    processed_qpos = qpos_array.copy()
    dof_names = qpos_dof_names(model)
    arm_joint_names = {
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    }
    arm_qpos_indices = [7 + index for index, name in enumerate(dof_names) if name in arm_joint_names]

    for qpos_index in arm_qpos_indices:
        values = processed_qpos[:, qpos_index].copy()
        if arm_smooth_window > 1:
            values = smooth_1d(values, arm_smooth_window)
        if max_arm_joint_step is not None and max_arm_joint_step > 0.0 and len(values) > 1:
            max_step = float(max_arm_joint_step)
            limited = values.copy()
            for frame_id in range(1, len(limited)):
                delta = np.clip(limited[frame_id] - limited[frame_id - 1], -max_step, max_step)
                limited[frame_id] = limited[frame_id - 1] + delta
            for frame_id in range(len(limited) - 2, -1, -1):
                delta = np.clip(limited[frame_id] - limited[frame_id + 1], -max_step, max_step)
                limited[frame_id] = limited[frame_id + 1] + delta
            values = limited
        processed_qpos[:, qpos_index] = values

    clip_joint_limits(model, processed_qpos)
    return processed_qpos


def build_lens110_motion(
    bvh_path,
    pkl_path,
    txt_path,
    target_xml,
    bvh_format,
    target_fps,
    max_source_frames,
    skip_initial_frames,
    method,
    hybrid_iterations,
    stance_width_offset,
    hip_yaw_out_offset,
    leg_motion_scale,
    hip_yaw_scale,
    ankle_roll_scale,
    torso_yaw_scale,
    lower_body_smooth_window,
    min_stance_width,
    stance_correction_gain,
    stance_correction_iterations,
    max_stance_correction_step,
    ground_mode,
    ground_clearance,
    foot_ground_offset,
    ground_percentile,
    max_ground_lowering,
    ground_smooth_window,
    root_z_offset,
    root_xy_scale,
    root_roll_scale,
    root_pitch_scale,
    max_lateral_roll,
    max_forward_pitch,
    max_backward_pitch,
    root_rot_smooth_window,
    root_z_smooth_window,
    root_z_motion_scale,
    post_lower_floating_feet,
    max_post_ground_lowering,
    arm_smooth_window,
    max_arm_joint_step,
    verbose,
):
    source_qpos, output_fps = retarget_bvh_to_g1_qpos(
        bvh_path,
        bvh_format,
        target_fps,
        max_source_frames,
        skip_initial_frames,
        verbose=verbose,
    )

    source_model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
    target_model = mj.MjModel.from_xml_path(str(target_xml))

    if method == "direct":
        qpos_array = direct_map_g1_to_lens110(
            source_model,
            target_model,
            source_qpos,
            stance_width_offset,
            hip_yaw_out_offset,
        )
    elif method == "hybrid":
        qpos_array = hybrid_map_g1_to_lens110(
            source_model,
            target_model,
            source_qpos,
            hybrid_iterations,
            stance_width_offset,
            hip_yaw_out_offset,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    dof_names = qpos_dof_names(target_model)
    qpos_array = apply_lens110_motion_postprocess(
        target_model,
        qpos_array,
        leg_motion_scale,
        hip_yaw_scale,
        ankle_roll_scale,
        torso_yaw_scale,
        lower_body_smooth_window,
    )
    qpos_array = apply_arm_postprocess(
        target_model,
        qpos_array,
        arm_smooth_window,
        max_arm_joint_step,
    )
    qpos_array = apply_root_upright_postprocess(
        qpos_array,
        root_roll_scale,
        root_pitch_scale,
        max_lateral_roll,
        max_forward_pitch,
        max_backward_pitch,
        root_rot_smooth_window,
    )
    qpos_array, stance_stats = enforce_min_stance_width(
        target_model,
        qpos_array,
        min_stance_width,
        stance_correction_gain,
        stance_correction_iterations,
        max_stance_correction_step,
    )
    ground_stats = None
    if ground_mode != "off":
        qpos_array, ground_stats = ground_qpos_by_feet(
            target_model,
            qpos_array,
            ("left_ankle_roll_link", "right_ankle_roll_link"),
            ground_clearance,
            foot_ground_offset,
            ground_mode,
            ground_percentile,
            max_ground_lowering,
            ground_smooth_window,
        )
    if root_z_offset != 0.0:
        qpos_array[:, 2] += root_z_offset
    qpos_array = apply_root_z_postprocess(qpos_array, root_z_smooth_window, root_z_motion_scale)
    if post_lower_floating_feet:
        qpos_array = lower_floating_feet_only(
            target_model,
            qpos_array,
            ("left_ankle_roll_link", "right_ankle_roll_link"),
            ground_clearance,
            foot_ground_offset,
            max_post_ground_lowering,
        )
    qpos_array = apply_root_xy_scale(qpos_array, root_xy_scale)

    motion_data = {
        "fps": float(output_fps),
        "root_pos": qpos_array[:, :3],
        "root_rot": qpos_array[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos_array[:, 7:],
        "dof_names": dof_names,
        "local_body_pos": None,
        "link_body_list": None,
        "source_file": str(bvh_path),
        "source_robot": "bvh_nokov_to_unitree_g1",
        "target_model_file": str(target_xml),
        "retarget_method": f"bvh_{bvh_format}->unitree_g1->{method}->lens110_21dof",
        "skip_initial_frames": int(skip_initial_frames),
        "root_upright_postprocess": {
            "root_roll_scale": float(root_roll_scale),
            "root_pitch_scale": float(root_pitch_scale),
            "max_lateral_roll": float(max_lateral_roll),
            "max_forward_pitch": float(max_forward_pitch),
            "max_backward_pitch": float(max_backward_pitch),
            "root_rot_smooth_window": int(root_rot_smooth_window),
        },
        "root_z_postprocess": {
            "root_z_smooth_window": int(root_z_smooth_window),
            "root_z_motion_scale": float(root_z_motion_scale),
            "post_lower_floating_feet": bool(post_lower_floating_feet),
            "max_post_ground_lowering": None
            if max_post_ground_lowering is None
            else float(max_post_ground_lowering),
        },
        "arm_postprocess": {
            "arm_smooth_window": int(arm_smooth_window),
            "max_arm_joint_step": None if max_arm_joint_step is None else float(max_arm_joint_step),
        },
    }

    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as file:
        pickle.dump(motion_data, file)
    convert_lens110_gmr_to_legged_lab(str(pkl_path), str(txt_path), fps=output_fps)

    return {
        "frames": int(len(qpos_array)),
        "fps": float(output_fps),
        "ground": ground_stats,
        "stance": stance_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch retarget BVH dance motions to Lens110.")
    parser.add_argument(
        "--src_dir",
        type=pathlib.Path,
        default=repository_root() / "datasets" / "dance" / "raw_bvh" / "original_bvh",
    )
    parser.add_argument(
        "--out_dir",
        type=pathlib.Path,
        default=repository_root() / "projects" / "01_dance_whole_body" / "data" / "processed" / "from_bvh",
    )
    parser.add_argument(
        "--target_xml",
        type=pathlib.Path,
        default=lens110_mjcf(),
    )
    parser.add_argument("--pattern", type=str, default="*.bvh")
    parser.add_argument("--bvh_format", choices=["lafan1", "nokov"], default="nokov")
    parser.add_argument("--target_fps", type=float, default=30.0)
    parser.add_argument("--max_source_frames", type=int, default=None)
    parser.add_argument("--skip_initial_frames", type=int, default=1)
    parser.add_argument("--method", choices=["direct", "hybrid"], default="hybrid")
    parser.add_argument("--hybrid_iterations", type=int, default=4)
    parser.add_argument("--stance_width_offset", type=float, default=0.10)
    parser.add_argument("--hip_yaw_out_offset", type=float, default=0.0)
    parser.add_argument("--leg_motion_scale", type=float, default=1.0)
    parser.add_argument("--hip_yaw_scale", type=float, default=1.0)
    parser.add_argument("--ankle_roll_scale", type=float, default=1.0)
    parser.add_argument("--torso_yaw_scale", type=float, default=1.0)
    parser.add_argument("--lower_body_smooth_window", type=int, default=3)
    parser.add_argument("--min_stance_width", type=float, default=0.20)
    parser.add_argument("--stance_correction_gain", type=float, default=3.5)
    parser.add_argument("--stance_correction_iterations", type=int, default=6)
    parser.add_argument("--max_stance_correction_step", type=float, default=0.16)
    parser.add_argument("--ground_mode", choices=["constant", "per_frame", "smooth", "off"], default="per_frame")
    parser.add_argument("--ground_clearance", type=float, default=0.0)
    parser.add_argument("--foot_ground_offset", type=float, default=0.0)
    parser.add_argument("--ground_percentile", type=float, default=2.0)
    parser.add_argument("--max_ground_lowering", type=float, default=0.20)
    parser.add_argument("--ground_smooth_window", type=int, default=7)
    parser.add_argument("--root_z_offset", type=float, default=0.0)
    parser.add_argument("--root_xy_scale", type=float, default=0.72)
    parser.add_argument("--root_roll_scale", type=float, default=1.0)
    parser.add_argument("--root_pitch_scale", type=float, default=1.0)
    parser.add_argument(
        "--max_lateral_roll",
        type=float,
        default=0.0,
        help="Clamp root side roll in radians. 0 disables the clamp.",
    )
    parser.add_argument(
        "--max_forward_pitch",
        type=float,
        default=0.0,
        help="Clamp root forward pitch in radians. 0 disables the clamp.",
    )
    parser.add_argument(
        "--max_backward_pitch",
        type=float,
        default=0.0,
        help="Clamp root backward pitch in radians. 0 disables the clamp.",
    )
    parser.add_argument("--root_rot_smooth_window", type=int, default=1)
    parser.add_argument(
        "--root_z_smooth_window",
        type=int,
        default=1,
        help="Moving-average window for root z after grounding. Use with --root_z_motion_scale to reduce body bobbing.",
    )
    parser.add_argument(
        "--root_z_motion_scale",
        type=float,
        default=1.0,
        help="Scale root-z motion around its median after optional smoothing. 1 keeps the motion.",
    )
    parser.add_argument("--post_lower_floating_feet", action="store_true", default=False)
    parser.add_argument("--max_post_ground_lowering", type=float, default=None)
    parser.add_argument("--arm_smooth_window", type=int, default=1)
    parser.add_argument("--max_arm_joint_step", type=float, default=None)
    parser.add_argument("--override", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    bvh_files = sorted(args.src_dir.glob(args.pattern))
    print(f"Found {len(bvh_files)} BVH files")
    pkl_dir = args.out_dir / "pkl"
    txt_dir = args.out_dir / "txt"
    used_stems = set()

    for index, bvh_path in enumerate(bvh_files, start=1):
        stem = sanitize_stem(bvh_path.stem)
        original_stem = stem
        suffix = 2
        while stem in used_stems:
            stem = f"{original_stem}_{suffix}"
            suffix += 1
        used_stems.add(stem)

        pkl_path = pkl_dir / f"{stem}.pkl"
        txt_path = txt_dir / f"{stem}.txt"
        if txt_path.exists() and not args.override:
            print(f"[{index}/{len(bvh_files)}] skip {bvh_path.name}")
            continue

        print(f"[{index}/{len(bvh_files)}] retarget {bvh_path.name} -> {stem}")
        stats = build_lens110_motion(
            bvh_path,
            pkl_path,
            txt_path,
            args.target_xml,
            args.bvh_format,
            args.target_fps,
            args.max_source_frames,
            args.skip_initial_frames,
            args.method,
            args.hybrid_iterations,
            args.stance_width_offset,
            args.hip_yaw_out_offset,
            args.leg_motion_scale,
            args.hip_yaw_scale,
            args.ankle_roll_scale,
            args.torso_yaw_scale,
            args.lower_body_smooth_window,
            args.min_stance_width,
            args.stance_correction_gain,
            args.stance_correction_iterations,
            args.max_stance_correction_step,
            args.ground_mode,
            args.ground_clearance,
            args.foot_ground_offset,
            args.ground_percentile,
            args.max_ground_lowering,
            args.ground_smooth_window,
            args.root_z_offset,
            args.root_xy_scale,
            args.root_roll_scale,
            args.root_pitch_scale,
            args.max_lateral_roll,
            args.max_forward_pitch,
            args.max_backward_pitch,
            args.root_rot_smooth_window,
            args.root_z_smooth_window,
            args.root_z_motion_scale,
            args.post_lower_floating_feet,
            args.max_post_ground_lowering,
            args.arm_smooth_window,
            args.max_arm_joint_step,
            args.verbose,
        )
        stance = stats["stance"]
        print(
            "  output: "
            f"frames={stats['frames']}, fps={stats['fps']:.2f}, "
            f"stance p5/mean={stance['after_p5']:.3f}/{stance['after_mean']:.3f}"
        )

    print(f"Done: {args.out_dir}")


if __name__ == "__main__":
    main()
