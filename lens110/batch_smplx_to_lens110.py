import argparse
import pathlib
import pickle
import sys

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast

HERE = pathlib.Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lens110_gmr_to_legged_lab import convert_lens110_gmr_to_legged_lab


LENS110_SCALE_TARGETS = {
    "head": "head",
    "left_hip": "left_hip_roll_link",
    "right_hip": "right_hip_roll_link",
    "left_knee": "left_knee_link",
    "right_knee": "right_knee_link",
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
    "left_shoulder": "left_shoulder_roll_link",
    "right_shoulder": "right_shoulder_roll_link",
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",
    "left_wrist": "left_hand",
    "right_wrist": "right_hand",
}


def get_qpos_dof_names(model):
    return [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_qposadr[joint_id] >= 7
    ]


def clip_qpos_joint_limits(model, qpos_array):
    for joint_id in range(model.njnt):
        qpos_addr = model.jnt_qposadr[joint_id]
        if qpos_addr < 7 or not model.jnt_limited[joint_id]:
            continue
        lower, upper = model.jnt_range[joint_id]
        qpos_array[:, qpos_addr] = np.clip(qpos_array[:, qpos_addr], lower, upper)


def apply_lens110_postprocess(
    model,
    qpos_array,
    knee_bend_offset,
    hip_pitch_offset,
    ankle_pitch_offset,
    torso_yaw_scale,
):
    dof_names = get_qpos_dof_names(model)
    processed_qpos = qpos_array.copy()
    for side in ("left", "right"):
        for joint_name, offset in (
            (f"{side}_knee_joint", knee_bend_offset),
            (f"{side}_hip_pitch_joint", hip_pitch_offset),
            (f"{side}_ankle_pitch_joint", ankle_pitch_offset),
        ):
            if joint_name in dof_names and offset != 0.0:
                processed_qpos[:, 7 + dof_names.index(joint_name)] += offset

    if "torso_yaw_joint" in dof_names and torso_yaw_scale != 1.0:
        processed_qpos[:, 7 + dof_names.index("torso_yaw_joint")] *= torso_yaw_scale

    clip_qpos_joint_limits(model, processed_qpos)
    return processed_qpos


def apply_root_pitch_offset(qpos_array, root_pitch_offset):
    if root_pitch_offset == 0.0:
        return qpos_array
    processed_qpos = qpos_array.copy()
    root_rot = R.from_quat(processed_qpos[:, 3:7][:, [1, 2, 3, 0]])
    corrected_root_rot = root_rot * R.from_euler("y", root_pitch_offset)
    processed_qpos[:, 3:7] = corrected_root_rot.as_quat()[:, [3, 0, 1, 2]]
    return processed_qpos


def moving_average(array, window):
    window = max(1, int(window))
    if window <= 1:
        return array.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / window
    if array.ndim == 1:
        padded = np.pad(array, (pad, pad), mode="edge")
        return np.convolve(padded, kernel, mode="valid")
    return np.stack([moving_average(array[:, index], window) for index in range(array.shape[1])], axis=1)


def apply_root_xy_postprocess(
    qpos_array,
    root_xy_scale,
    root_lateral_scale,
    root_xy_smooth_window,
    max_root_xy_step,
):
    if (
        root_xy_scale == 1.0
        and root_lateral_scale == 1.0
        and root_xy_smooth_window <= 1
        and max_root_xy_step is None
    ):
        return qpos_array

    processed_qpos = qpos_array.copy()
    xy = processed_qpos[:, :2].copy()
    xy = xy[0] + root_xy_scale * (xy - xy[0])
    if root_lateral_scale != 1.0 and len(xy) > 1:
        displacement = xy[-1] - xy[0]
        displacement_norm = np.linalg.norm(displacement)
        if displacement_norm > 1e-6:
            forward = displacement / displacement_norm
            lateral = np.array([-forward[1], forward[0]], dtype=np.float64)
            offset = xy - xy[0]
            forward_offset = np.outer(offset @ forward, forward)
            lateral_offset = np.outer(offset @ lateral, lateral)
            xy = xy[0] + forward_offset + root_lateral_scale * lateral_offset

    if root_xy_smooth_window > 1:
        smoothed_xy = moving_average(xy, root_xy_smooth_window)
        xy = smoothed_xy - smoothed_xy[0] + xy[0]

    if max_root_xy_step is not None and max_root_xy_step > 0.0:
        limited_xy = xy.copy()
        for frame_id in range(1, len(limited_xy)):
            delta = xy[frame_id] - limited_xy[frame_id - 1]
            delta_norm = np.linalg.norm(delta)
            if delta_norm > max_root_xy_step:
                delta = delta / delta_norm * max_root_xy_step
            limited_xy[frame_id] = limited_xy[frame_id - 1] + delta
        xy = limited_xy

    processed_qpos[:, :2] = xy
    return processed_qpos


def apply_root_rotation_postprocess(qpos_array, root_roll_scale, root_pitch_scale, root_rot_smooth_window):
    if root_roll_scale == 1.0 and root_pitch_scale == 1.0 and root_rot_smooth_window <= 1:
        return qpos_array
    processed_qpos = qpos_array.copy()
    euler = R.from_quat(processed_qpos[:, 3:7][:, [1, 2, 3, 0]]).as_euler("XYZ")
    euler = np.unwrap(euler, axis=0)
    if root_rot_smooth_window > 1:
        euler = moving_average(euler, root_rot_smooth_window)
    mean_roll = np.mean(euler[:, 0])
    mean_pitch = np.mean(euler[:, 1])
    euler[:, 0] = mean_roll + root_roll_scale * (euler[:, 0] - mean_roll)
    euler[:, 1] = mean_pitch + root_pitch_scale * (euler[:, 1] - mean_pitch)
    processed_qpos[:, 3:7] = R.from_euler("XYZ", euler).as_quat()[:, [3, 0, 1, 2]]
    return processed_qpos


def apply_root_z_postprocess(qpos_array, root_z_smooth_window, root_z_motion_scale):
    if root_z_smooth_window <= 1 and root_z_motion_scale == 1.0:
        return qpos_array
    processed_qpos = qpos_array.copy()
    root_z = processed_qpos[:, 2]
    smoothed_root_z = moving_average(root_z, root_z_smooth_window)
    mean_root_z = np.mean(smoothed_root_z)
    processed_qpos[:, 2] = mean_root_z + root_z_motion_scale * (smoothed_root_z - mean_root_z)
    return processed_qpos


def get_body_distance_from_pelvis(model, body_name):
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    pelvis_pos = data.body("pelvis").xpos.copy()
    return float(np.linalg.norm(data.body(body_name).xpos.copy() - pelvis_pos))


def get_smplx_distance_from_pelvis(smplx_output, body_model, body_name):
    joint_names = JOINT_NAMES[: len(body_model.parents)]
    joint_indices = {name: index for index, name in enumerate(joint_names)}
    if body_name not in joint_indices:
        raise ValueError(f"SMPL-X body not found: {body_name}")
    joints = smplx_output.joints.detach().cpu().numpy()
    pelvis = joints[:, joint_indices["pelvis"]]
    body = joints[:, joint_indices[body_name]]
    distances = np.linalg.norm(body - pelvis, axis=1)
    return float(np.mean(distances))


def compute_lens110_body_scale_table(
    model,
    smplx_output,
    body_model,
    root_translation_scale,
    min_body_scale,
    max_body_scale,
):
    scale_table = {}
    raw_scales = {}
    for human_body, robot_body in LENS110_SCALE_TARGETS.items():
        robot_distance = get_body_distance_from_pelvis(model, robot_body)
        human_distance = get_smplx_distance_from_pelvis(smplx_output, body_model, human_body)
        if human_distance <= 1e-6:
            continue
        raw_scale = robot_distance / human_distance
        scale = float(np.clip(raw_scale, min_body_scale, max_body_scale))
        scale_table[human_body] = scale
        raw_scales[human_body] = raw_scale

    lower_body_scales = [
        scale_table[name]
        for name in ("left_knee", "right_knee", "left_foot", "right_foot")
        if name in scale_table
    ]
    if lower_body_scales:
        root_scale = float(np.median(lower_body_scales) * root_translation_scale)
    else:
        root_scale = root_translation_scale
    scale_table["pelvis"] = float(np.clip(root_scale, min_body_scale, max_body_scale))
    raw_scales["pelvis"] = root_scale
    return scale_table, raw_scales


def apply_arm_postprocess(model, qpos_array, arm_swing_scale, arm_smooth_window):
    if arm_swing_scale == 1.0 and arm_smooth_window <= 1:
        return qpos_array
    dof_names = get_qpos_dof_names(model)
    neutral_arm_pose = {
        "left_shoulder_pitch_joint": 0.4,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": -0.8,
        "right_shoulder_pitch_joint": 0.4,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": -0.8,
    }
    # Preserve sagittal-plane arm swing and elbow flexion more aggressively,
    # while damping roll/yaw noise that tends to look erratic on lens110.
    arm_damping = {
        "left_shoulder_pitch_joint": 0.25,
        "right_shoulder_pitch_joint": 0.25,
        "left_elbow_joint": 0.45,
        "right_elbow_joint": 0.45,
        "left_shoulder_roll_joint": 1.0,
        "right_shoulder_roll_joint": 1.0,
        "left_shoulder_yaw_joint": 0.9,
        "right_shoulder_yaw_joint": 0.9,
    }
    processed_qpos = qpos_array.copy()
    for joint_name, neutral_value in neutral_arm_pose.items():
        if joint_name not in dof_names:
            continue
        qpos_index = 7 + dof_names.index(joint_name)
        values = moving_average(processed_qpos[:, qpos_index], arm_smooth_window)
        damping = arm_damping.get(joint_name, 1.0)
        effective_scale = 1.0 - (1.0 - arm_swing_scale) * damping
        processed_qpos[:, qpos_index] = neutral_value + effective_scale * (values - neutral_value)
    clip_qpos_joint_limits(model, processed_qpos)
    return processed_qpos


def apply_roll_joint_postprocess(model, qpos_array, hip_roll_scale, ankle_roll_scale, roll_smooth_window):
    if hip_roll_scale == 1.0 and ankle_roll_scale == 1.0 and roll_smooth_window <= 1:
        return qpos_array
    dof_names = get_qpos_dof_names(model)
    processed_qpos = qpos_array.copy()
    for side in ("left", "right"):
        for joint_name, scale in (
            (f"{side}_hip_roll_joint", hip_roll_scale),
            (f"{side}_ankle_roll_joint", ankle_roll_scale),
        ):
            if joint_name not in dof_names:
                continue
            qpos_index = 7 + dof_names.index(joint_name)
            values = processed_qpos[:, qpos_index]
            if roll_smooth_window > 1:
                values = moving_average(values, roll_smooth_window)
            mean_value = np.mean(values)
            processed_qpos[:, qpos_index] = mean_value + scale * (values - mean_value)
    clip_qpos_joint_limits(model, processed_qpos)
    return processed_qpos


def get_mesh_geom_min_z(model, data, geom_id):
    mesh_id = model.geom_dataid[geom_id]
    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[start : start + count]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    world_vertices = data.geom_xpos[geom_id] + vertices @ rotation.T
    return world_vertices[:, 2].min()


def get_foot_mesh_min_z(model, qpos_array, foot_body_names):
    data = mj.MjData(model)
    foot_body_ids = {model.body(name).id for name in foot_body_names}
    foot_geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] in foot_body_ids and model.geom_type[geom_id] == mj.mjtGeom.mjGEOM_MESH
    ]
    if not foot_geom_ids:
        raise ValueError(f"No foot mesh geoms found for bodies: {foot_body_names}")

    foot_min_z = []
    for frame_id in range(len(qpos_array)):
        data.qpos[:] = qpos_array[frame_id]
        mj.mj_forward(model, data)
        foot_min_z.append(min(get_mesh_geom_min_z(model, data, geom_id) for geom_id in foot_geom_ids))
    return np.asarray(foot_min_z)


def ground_qpos_by_feet(
    model,
    qpos_array,
    foot_body_names,
    ground_clearance,
    foot_ground_offset,
    ground_mode,
    ground_percentile,
    max_ground_lowering,
    ground_smooth_window,
):
    foot_min_z = get_foot_mesh_min_z(model, qpos_array, foot_body_names)
    foot_min_z = foot_min_z - foot_ground_offset
    grounded_qpos = qpos_array.copy()
    offset = 0.0
    if ground_mode == "per_frame":
        offsets = foot_min_z - ground_clearance
        if max_ground_lowering is not None:
            offsets = np.minimum(offsets, max_ground_lowering)
        grounded_qpos[:, 2] -= offsets
    elif ground_mode == "smooth":
        offsets = foot_min_z - ground_clearance
        if max_ground_lowering is not None:
            offsets = np.minimum(offsets, max_ground_lowering)
        window = max(1, int(ground_smooth_window))
        if window > 1:
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded_offsets = np.pad(offsets, (pad_left, pad_right), mode="edge")
            kernel = np.ones(window, dtype=np.float64) / window
            offsets = np.convolve(padded_offsets, kernel, mode="valid")
        grounded_qpos[:, 2] -= offsets
    elif ground_mode == "constant":
        offset = np.percentile(foot_min_z, ground_percentile) - ground_clearance
        if max_ground_lowering is not None:
            offset = min(offset, max_ground_lowering)
        grounded_qpos[:, 2] -= offset
    stats = {
        "foot_min_before": float(foot_min_z.min()),
        "foot_p5_before": float(np.percentile(foot_min_z, 5)),
        "foot_mean_before": float(foot_min_z.mean()),
        "ground_offset": float(offset),
    }
    return grounded_qpos, stats


def retarget_file(
    smplx_file,
    pkl_path,
    txt_path,
    smplx_folder,
    target_fps,
    ground_mode,
    ground_clearance,
    ground_percentile,
    max_ground_lowering,
    root_z_offset,
    ground_smooth_window,
    scale_mode,
    root_translation_scale,
    min_body_scale,
    max_body_scale,
    root_xy_scale,
    root_lateral_scale,
    root_xy_smooth_window,
    max_root_xy_step,
    root_roll_scale,
    root_pitch_scale,
    root_rot_smooth_window,
    hip_roll_scale,
    ankle_roll_scale,
    roll_smooth_window,
    knee_bend_offset,
    hip_pitch_offset,
    ankle_pitch_offset,
    torso_yaw_scale,
    root_pitch_offset,
    root_z_smooth_window,
    root_z_motion_scale,
    arm_swing_scale,
    arm_smooth_window,
):
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(smplx_file, smplx_folder)
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data,
        body_model,
        smplx_output,
        tgt_fps=target_fps,
    )
    retargeter = GMR(
        src_human="smplx",
        tgt_robot="lens110_21dof",
        actual_human_height=actual_human_height,
        verbose=False,
    )
    if scale_mode == "lens110_body":
        scale_table, raw_scales = compute_lens110_body_scale_table(
            retargeter.model,
            smplx_output,
            body_model,
            root_translation_scale,
            min_body_scale,
            max_body_scale,
        )
        retargeter.human_scale_table.update(scale_table)
        print(
            "  lens110_body_scale: "
            + ", ".join(
                f"{name}={scale_table[name]:.3f}"
                + ("" if abs(scale_table[name] - raw_scales[name]) < 1e-6 else f"(raw {raw_scales[name]:.3f})")
                for name in sorted(scale_table)
            )
        )

    qpos_array = np.asarray([retargeter.retarget(frame).copy() for frame in frames])
    qpos_array = apply_root_xy_postprocess(
        qpos_array,
        root_xy_scale,
        root_lateral_scale,
        root_xy_smooth_window,
        max_root_xy_step,
    )
    qpos_array = apply_lens110_postprocess(
        retargeter.model,
        qpos_array,
        knee_bend_offset,
        hip_pitch_offset,
        ankle_pitch_offset,
        torso_yaw_scale,
    )
    qpos_array = apply_root_pitch_offset(qpos_array, root_pitch_offset)
    qpos_array = apply_root_rotation_postprocess(
        qpos_array,
        root_roll_scale,
        root_pitch_scale,
        root_rot_smooth_window,
    )
    qpos_array = apply_root_z_postprocess(qpos_array, root_z_smooth_window, root_z_motion_scale)
    qpos_array = apply_arm_postprocess(retargeter.model, qpos_array, arm_swing_scale, arm_smooth_window)
    qpos_array = apply_roll_joint_postprocess(
        retargeter.model,
        qpos_array,
        hip_roll_scale,
        ankle_roll_scale,
        roll_smooth_window,
    )
    ground_stats = None
    if ground_mode != "off":
        qpos_array, ground_stats = ground_qpos_by_feet(
            retargeter.model,
            qpos_array,
            ("left_ankle_roll_link", "right_ankle_roll_link"),
            ground_clearance,
            0.0,
            ground_mode,
            ground_percentile,
            max_ground_lowering,
            ground_smooth_window,
        )
    if root_z_offset != 0.0:
        qpos_array[:, 2] += root_z_offset
    motion_data = {
        "fps": aligned_fps,
        "root_pos": qpos_array[:, :3],
        "root_rot": qpos_array[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos_array[:, 7:],
        "dof_names": get_qpos_dof_names(retargeter.model),
        "local_body_pos": None,
        "link_body_list": None,
    }

    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as file:
        pickle.dump(motion_data, file)
    convert_lens110_gmr_to_legged_lab(str(pkl_path), str(txt_path), fps=aligned_fps)
    if ground_stats is not None:
        print(
            "  grounding: "
            f"mode={ground_mode}, "
            f"foot_min/p5/mean(before)="
            f"{ground_stats['foot_min_before']:.3f}/"
            f"{ground_stats['foot_p5_before']:.3f}/"
            f"{ground_stats['foot_mean_before']:.3f}, "
            f"offset={ground_stats['ground_offset']:.3f}, "
            f"root_z_offset={root_z_offset:.3f}, "
            f"root_pitch_offset={root_pitch_offset:.3f}, "
            f"root_z_smooth_window={root_z_smooth_window}, "
            f"root_z_motion_scale={root_z_motion_scale:.3f}, "
            f"knee_offset={knee_bend_offset:.3f}, "
            f"arm_swing_scale={arm_swing_scale:.3f}, "
            f"torso_yaw_scale={torso_yaw_scale:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=pathlib.Path, required=True)
    parser.add_argument("--out_dir", type=pathlib.Path, required=True)
    parser.add_argument("--target_fps", type=int, default=30)
    parser.add_argument("--override", action="store_true", default=False)
    parser.add_argument("--ground_mode", choices=["constant", "per_frame", "smooth", "off"], default="per_frame")
    parser.add_argument("--ground_clearance", type=float, default=0.0)
    parser.add_argument("--ground_percentile", type=float, default=5.0)
    parser.add_argument("--ground_smooth_window", type=int, default=5)
    parser.add_argument(
        "--scale_mode",
        choices=["legacy", "lens110_body"],
        default="legacy",
        help="legacy uses GMR's single human-height scale; lens110_body fits SMPL-X target distances to Lens110 body distances.",
    )
    parser.add_argument(
        "--root_translation_scale",
        type=float,
        default=1.0,
        help="Extra multiplier for root translation when --scale_mode lens110_body is used.",
    )
    parser.add_argument("--min_body_scale", type=float, default=0.45)
    parser.add_argument("--max_body_scale", type=float, default=1.35)
    parser.add_argument(
        "--root_xy_scale",
        type=float,
        default=1.0,
        help="Scale root horizontal trajectory around the first frame after IK.",
    )
    parser.add_argument(
        "--root_lateral_scale",
        type=float,
        default=1.0,
        help="Scale root horizontal motion perpendicular to the net walking direction.",
    )
    parser.add_argument(
        "--root_xy_smooth_window",
        type=int,
        default=1,
        help="Moving-average window for root horizontal trajectory after IK.",
    )
    parser.add_argument(
        "--max_root_xy_step",
        type=float,
        default=None,
        help="Optional maximum root horizontal displacement per output frame in meters.",
    )
    parser.add_argument("--root_roll_scale", type=float, default=1.0)
    parser.add_argument("--root_pitch_scale", type=float, default=1.0)
    parser.add_argument("--root_rot_smooth_window", type=int, default=1)
    parser.add_argument("--hip_roll_scale", type=float, default=1.0)
    parser.add_argument("--ankle_roll_scale", type=float, default=1.0)
    parser.add_argument("--roll_smooth_window", type=int, default=1)
    parser.add_argument(
        "--max_ground_lowering",
        type=float,
        default=0.25,
        help="Optional cap for downward root shift in meters; useful to avoid crouched motions.",
    )
    parser.add_argument(
        "--root_z_offset",
        type=float,
        default=0.0,
        help="Additional vertical root offset in meters after foot grounding.",
    )
    parser.add_argument("--knee_bend_offset", type=float, default=0.25)
    parser.add_argument("--hip_pitch_offset", type=float, default=-0.12)
    parser.add_argument("--ankle_pitch_offset", type=float, default=-0.12)
    parser.add_argument("--torso_yaw_scale", type=float, default=0.35)
    parser.add_argument(
        "--root_pitch_offset",
        type=float,
        default=0.06,
        help="Local root pitch correction in radians. Positive values reduce backward lean for lens110.",
    )
    parser.add_argument(
        "--root_z_smooth_window",
        type=int,
        default=1,
        help="Moving-average window for root z before grounding. Use 15-31 to reduce vertical popping.",
    )
    parser.add_argument(
        "--root_z_motion_scale",
        type=float,
        default=1.0,
        help="Scale root-z motion around its mean after smoothing. 1 keeps motion, 0 makes root height nearly constant.",
    )
    parser.add_argument(
        "--arm_swing_scale",
        type=float,
        default=1.0,
        help="Blend arm joints toward the default lens110 arm pose. Use 0.3-0.5 for noisy SMPL-X arms.",
    )
    parser.add_argument(
        "--arm_smooth_window",
        type=int,
        default=1,
        help="Moving-average window for arm joints.",
    )
    args = parser.parse_args()

    smplx_folder = HERE.parent / "assets" / "body_models"
    pkl_dir = args.out_dir / "pkl"
    txt_dir = args.out_dir / "txt"
    files = sorted(args.src_dir.glob("*.npz"))
    print(f"Found {len(files)} files")

    for index, smplx_file in enumerate(files, start=1):
        pkl_path = pkl_dir / f"{smplx_file.stem}.pkl"
        txt_path = txt_dir / f"{smplx_file.stem}.txt"
        if txt_path.exists() and not args.override:
            print(f"[{index}/{len(files)}] skip {smplx_file.name}")
            continue
        print(f"[{index}/{len(files)}] retarget {smplx_file.name}")
        retarget_file(
            smplx_file,
            pkl_path,
            txt_path,
            smplx_folder,
            args.target_fps,
            args.ground_mode,
            args.ground_clearance,
            args.ground_percentile,
            args.max_ground_lowering,
            args.root_z_offset,
            args.ground_smooth_window,
            args.scale_mode,
            args.root_translation_scale,
            args.min_body_scale,
            args.max_body_scale,
            args.root_xy_scale,
            args.root_lateral_scale,
            args.root_xy_smooth_window,
            args.max_root_xy_step,
            args.root_roll_scale,
            args.root_pitch_scale,
            args.root_rot_smooth_window,
            args.hip_roll_scale,
            args.ankle_roll_scale,
            args.roll_smooth_window,
            args.knee_bend_offset,
            args.hip_pitch_offset,
            args.ankle_pitch_offset,
            args.torso_yaw_scale,
            args.root_pitch_offset,
            args.root_z_smooth_window,
            args.root_z_motion_scale,
            args.arm_swing_scale,
            args.arm_smooth_window,
        )

    print(f"Done: {args.out_dir}")


if __name__ == "__main__":
    main()
