import argparse
import pathlib
import pickle
import sys

import mujoco as mj
import mink
import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT

HERE = pathlib.Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from batch_smplx_to_lens110 import get_qpos_dof_names, ground_qpos_by_feet
from lens110_gmr_to_legged_lab import convert_lens110_gmr_to_legged_lab
from path_utils import lens110_mjcf, repository_root  # noqa: E402


G1_BODY_NAMES = (
    "pelvis",
    "torso_link",
    "head_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_rubber_hand",
    "right_rubber_hand",
)

DIRECT_JOINT_NAME_OVERRIDES = {
    "torso_yaw_joint": "waist_yaw_joint",
}

DIRECT_JOINT_SIGNS = {
    "left_elbow_joint": -1.0,
    "right_elbow_joint": -1.0,
}

LENS110_NEUTRAL_DOF = {
    "left_hip_pitch_joint": -0.14,
    "left_hip_roll_joint": 0.01,
    "left_hip_yaw_joint": -0.10,
    "left_knee_joint": 0.36,
    "left_ankle_pitch_joint": -0.20,
    "left_ankle_roll_joint": -0.20,
    "right_hip_pitch_joint": -0.14,
    "right_hip_roll_joint": -0.01,
    "right_hip_yaw_joint": 0.10,
    "right_knee_joint": 0.36,
    "right_ankle_pitch_joint": -0.20,
    "right_ankle_roll_joint": -0.20,
    "torso_yaw_joint": 0.0,
}


def normalize_output_suffix(suffix):
    if not suffix:
        return ""
    return suffix if suffix.startswith("_") else f"_{suffix}"


def apply_root_xy_scale(qpos_array, root_xy_scale):
    if root_xy_scale == 1.0:
        return qpos_array
    processed_qpos = qpos_array.copy()
    root_xy = processed_qpos[:, :2]
    root_xy_origin = root_xy[:1]
    processed_qpos[:, :2] = root_xy_origin + root_xy_scale * (root_xy - root_xy_origin)
    return processed_qpos


def moving_average(array, window):
    window = max(1, int(window))
    if window <= 1:
        return array.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(array, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def apply_lens110_motion_postprocess(
    target_model,
    qpos_array,
    leg_motion_scale,
    hip_yaw_scale,
    ankle_roll_scale,
    torso_yaw_scale,
    lower_body_smooth_window,
):
    if (
        leg_motion_scale == 1.0
        and hip_yaw_scale == 1.0
        and ankle_roll_scale == 1.0
        and torso_yaw_scale == 1.0
        and lower_body_smooth_window <= 1
    ):
        return qpos_array

    processed_qpos = qpos_array.copy()
    dof_names = qpos_dof_names(target_model)
    dof_index = {name: 7 + idx for idx, name in enumerate(dof_names)}

    joint_scales = {
        "left_hip_pitch_joint": leg_motion_scale,
        "right_hip_pitch_joint": leg_motion_scale,
        "left_hip_roll_joint": leg_motion_scale,
        "right_hip_roll_joint": leg_motion_scale,
        "left_hip_yaw_joint": hip_yaw_scale,
        "right_hip_yaw_joint": hip_yaw_scale,
        "left_knee_joint": leg_motion_scale,
        "right_knee_joint": leg_motion_scale,
        "left_ankle_pitch_joint": leg_motion_scale,
        "right_ankle_pitch_joint": leg_motion_scale,
        "left_ankle_roll_joint": ankle_roll_scale,
        "right_ankle_roll_joint": ankle_roll_scale,
        "torso_yaw_joint": torso_yaw_scale,
    }

    for joint_name, neutral_value in LENS110_NEUTRAL_DOF.items():
        if joint_name not in dof_index:
            continue
        qpos_index = dof_index[joint_name]
        values = moving_average(processed_qpos[:, qpos_index], lower_body_smooth_window)
        scale = joint_scales.get(joint_name, 1.0)
        processed_qpos[:, qpos_index] = neutral_value + scale * (values - neutral_value)

    clip_lens110_joint_limits(processed_qpos, target_model, dof_names)
    return processed_qpos


def load_g1_csv_qpos(csv_path, model):
    values = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != model.nq:
        raise ValueError(f"{csv_path} has {values.shape[1]} columns, expected {model.nq}")
    qpos = values.copy()
    qpos[:, 3:7] = values[:, 3:7][:, [3, 0, 1, 2]]
    return qpos


def g1_qpos_to_body_frames(model, qpos_array, body_names):
    data = mj.MjData(model)
    frames = []
    body_ids = {name: model.body(name).id for name in body_names}
    for qpos in qpos_array:
        data.qpos[:] = qpos
        mj.mj_forward(model, data)
        frames.append(
            {
                name: [data.xpos[body_id].copy(), data.xquat[body_id].copy()]
                for name, body_id in body_ids.items()
            }
        )
    return frames


def qpos_dof_names(model):
    return [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_qposadr[joint_id] >= 7
    ]


def quat_wxyz_to_rotation_matrix(quat_wxyz):
    w, x, y, z = quat_wxyz
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def apply_lens110_leg_adjustments(qpos_array, target_names, stance_width_offset, hip_yaw_out_offset):
    adjustments = {
        "left_hip_roll_joint": stance_width_offset,
        "right_hip_roll_joint": -stance_width_offset,
        "left_hip_yaw_joint": hip_yaw_out_offset,
        "right_hip_yaw_joint": -hip_yaw_out_offset,
    }
    for joint_name, offset in adjustments.items():
        if offset == 0.0 or joint_name not in target_names:
            continue
        qpos_array[:, 7 + target_names.index(joint_name)] += offset


def clip_lens110_joint_limits(qpos_array, target_model, target_names):
    for target_index, target_name in enumerate(target_names):
        target_joint_id = target_model.joint(target_name).id
        if target_model.jnt_limited[target_joint_id]:
            lower, upper = target_model.jnt_range[target_joint_id]
            qpos_array[:, 7 + target_index] = np.clip(qpos_array[:, 7 + target_index], lower, upper)


def enforce_min_stance_width(
    target_model,
    qpos_array,
    min_stance_width,
    stance_correction_gain,
    stance_correction_iterations,
    max_stance_correction_step,
):
    if min_stance_width <= 0.0 or len(qpos_array) == 0:
        return qpos_array, None

    processed_qpos = qpos_array.copy()
    data = mj.MjData(target_model)
    left_foot_id = target_model.body("left_ankle_roll_link").id
    right_foot_id = target_model.body("right_ankle_roll_link").id

    left_hip_roll_id = target_model.joint("left_hip_roll_joint").id
    right_hip_roll_id = target_model.joint("right_hip_roll_joint").id
    left_hip_roll_qpos = target_model.jnt_qposadr[left_hip_roll_id]
    right_hip_roll_qpos = target_model.jnt_qposadr[right_hip_roll_id]
    left_limits = target_model.jnt_range[left_hip_roll_id]
    right_limits = target_model.jnt_range[right_hip_roll_id]

    before_lateral = np.zeros(len(processed_qpos), dtype=np.float64)
    after_lateral = np.zeros(len(processed_qpos), dtype=np.float64)
    corrected_frames = 0
    total_hip_roll_adjustment = 0.0
    iterations = max(1, int(stance_correction_iterations))

    for frame_id in range(len(processed_qpos)):
        data.qpos[:] = processed_qpos[frame_id]
        mj.mj_forward(target_model, data)
        delta_world = data.xpos[left_foot_id].copy() - data.xpos[right_foot_id].copy()
        root_rotation = quat_wxyz_to_rotation_matrix(processed_qpos[frame_id, 3:7])
        delta_local = root_rotation.T @ delta_world
        before_lateral[frame_id] = abs(delta_local[1])

        frame_adjusted = False
        for _ in range(iterations):
            lateral_separation = abs(delta_local[1])
            deficit = min_stance_width - lateral_separation
            if deficit <= 1e-4:
                break

            hip_roll_delta = min(max_stance_correction_step, deficit * stance_correction_gain)
            new_left = np.clip(processed_qpos[frame_id, left_hip_roll_qpos] + hip_roll_delta, left_limits[0], left_limits[1])
            new_right = np.clip(
                processed_qpos[frame_id, right_hip_roll_qpos] - hip_roll_delta,
                right_limits[0],
                right_limits[1],
            )
            applied_delta = max(
                abs(new_left - processed_qpos[frame_id, left_hip_roll_qpos]),
                abs(new_right - processed_qpos[frame_id, right_hip_roll_qpos]),
            )
            if applied_delta <= 1e-6:
                break

            processed_qpos[frame_id, left_hip_roll_qpos] = new_left
            processed_qpos[frame_id, right_hip_roll_qpos] = new_right
            total_hip_roll_adjustment += applied_delta
            frame_adjusted = True

            data.qpos[:] = processed_qpos[frame_id]
            mj.mj_forward(target_model, data)
            delta_world = data.xpos[left_foot_id].copy() - data.xpos[right_foot_id].copy()
            root_rotation = quat_wxyz_to_rotation_matrix(processed_qpos[frame_id, 3:7])
            delta_local = root_rotation.T @ delta_world

        after_lateral[frame_id] = abs(delta_local[1])
        if frame_adjusted:
            corrected_frames += 1

    stats = {
        "before_min": float(before_lateral.min()),
        "before_p5": float(np.percentile(before_lateral, 5)),
        "before_mean": float(before_lateral.mean()),
        "after_min": float(after_lateral.min()),
        "after_p5": float(np.percentile(after_lateral, 5)),
        "after_mean": float(after_lateral.mean()),
        "corrected_frames": int(corrected_frames),
        "total_hip_roll_adjustment": float(total_hip_roll_adjustment),
    }
    return processed_qpos, stats


def direct_map_g1_to_lens110(
    source_model,
    target_model,
    source_qpos,
    stance_width_offset,
    hip_yaw_out_offset,
):
    source_names = qpos_dof_names(source_model)
    target_names = qpos_dof_names(target_model)
    source_index = {name: index for index, name in enumerate(source_names)}

    target_qpos = np.zeros((len(source_qpos), target_model.nq), dtype=np.float64)
    target_qpos[:, :7] = source_qpos[:, :7]
    for target_index, target_name in enumerate(target_names):
        source_name = DIRECT_JOINT_NAME_OVERRIDES.get(target_name, target_name)
        if source_name in source_index:
            sign = DIRECT_JOINT_SIGNS.get(target_name, 1.0)
            target_qpos[:, 7 + target_index] = sign * source_qpos[:, 7 + source_index[source_name]]

    apply_lens110_leg_adjustments(
        target_qpos,
        target_names,
        stance_width_offset,
        hip_yaw_out_offset,
    )
    clip_lens110_joint_limits(target_qpos, target_model, target_names)

    return target_qpos


def get_dof_indices_by_joint_names(model, joint_names):
    indices = []
    for dof_index in range(model.nv):
        joint_id = model.dof_jntid[dof_index]
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name in joint_names:
            indices.append(dof_index)
    return indices


def hybrid_map_g1_to_lens110(
    source_model,
    target_model,
    source_qpos,
    hybrid_iterations,
    stance_width_offset,
    hip_yaw_out_offset,
):
    qpos_array = direct_map_g1_to_lens110(
        source_model,
        target_model,
        source_qpos,
        stance_width_offset,
        hip_yaw_out_offset,
    )
    source_data = mj.MjData(source_model)
    configuration = mink.Configuration(target_model)

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
    arm_dof_indices = set(get_dof_indices_by_joint_names(target_model, arm_joint_names))
    frozen_dof_indices = [index for index in range(target_model.nv) if index not in arm_dof_indices]
    freeze_non_arm = mink.DofFreezingTask(target_model, frozen_dof_indices)
    limits = [mink.ConfigurationLimit(target_model)]

    task_pairs = (
        ("left_elbow_link", "left_elbow_link", 80.0),
        ("right_elbow_link", "right_elbow_link", 80.0),
        ("left_hand", "left_rubber_hand", 160.0),
        ("right_hand", "right_rubber_hand", 160.0),
    )
    tasks = [
        mink.FrameTask(
            frame_name=target_body,
            frame_type="body",
            position_cost=position_cost,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        for target_body, _, position_cost in task_pairs
    ]

    source_body_ids = {
        source_body: source_model.body(source_body).id
        for _, source_body, _ in task_pairs
    }
    dt = target_model.opt.timestep
    for frame_index, source_frame_qpos in enumerate(source_qpos):
        source_data.qpos[:] = source_frame_qpos
        mj.mj_forward(source_model, source_data)
        configuration.update(qpos_array[frame_index])
        for task, (_, source_body, _) in zip(tasks, task_pairs):
            source_body_id = source_body_ids[source_body]
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(source_data.xquat[source_body_id].copy()),
                    source_data.xpos[source_body_id].copy(),
                )
            )
        for _ in range(hybrid_iterations):
            velocity = mink.solve_ik(
                configuration,
                tasks,
                dt,
                solver="daqp",
                damping=5e-1,
                limits=limits,
                constraints=[freeze_non_arm],
            )
            configuration.integrate_inplace(velocity, dt)
        qpos_array[frame_index] = configuration.q.copy()

    return qpos_array


def retarget_file(
    csv_path,
    pkl_path,
    txt_path,
    target_xml,
    fps,
    ground_mode,
    ground_clearance,
    foot_ground_offset,
    ground_percentile,
    max_ground_lowering,
    root_z_offset,
    ground_smooth_window,
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
    root_xy_scale,
    frame_start,
    frame_end,
):
    source_model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
    source_qpos = load_g1_csv_qpos(csv_path, source_model)
    target_model_path = pathlib.Path(target_xml) if target_xml is not None else ROBOT_XML_DICT["lens110_21dof"]
    target_model = mj.MjModel.from_xml_path(str(target_model_path))

    if method == "direct":
        qpos_array = direct_map_g1_to_lens110(
            source_model,
            target_model,
            source_qpos,
            stance_width_offset,
            hip_yaw_out_offset,
        )
        dof_names = qpos_dof_names(target_model)
        ground_model = target_model
    elif method == "hybrid":
        qpos_array = hybrid_map_g1_to_lens110(
            source_model,
            target_model,
            source_qpos,
            hybrid_iterations,
            stance_width_offset,
            hip_yaw_out_offset,
        )
        dof_names = qpos_dof_names(target_model)
        ground_model = target_model
    else:
        source_frames = g1_qpos_to_body_frames(source_model, source_qpos, G1_BODY_NAMES)
        retargeter = GMR(
            src_human="unitree_g1",
            tgt_robot="lens110_21dof",
            actual_human_height=None,
            verbose=False,
        )
        qpos_array = np.asarray([retargeter.retarget(frame).copy() for frame in source_frames])
        dof_names = get_qpos_dof_names(retargeter.model)
        ground_model = retargeter.model

    qpos_array = apply_lens110_motion_postprocess(
        target_model,
        qpos_array,
        leg_motion_scale,
        hip_yaw_scale,
        ankle_roll_scale,
        torso_yaw_scale,
        lower_body_smooth_window,
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
            ground_model,
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
    qpos_array = apply_root_xy_scale(qpos_array, root_xy_scale)

    trim_start_frame = None
    trim_end_frame = None
    if frame_start is not None or frame_end is not None:
        trim_start_frame = 0 if frame_start is None else frame_start
        trim_end_frame = len(qpos_array) - 1 if frame_end is None else frame_end
        if trim_start_frame < 0 or trim_end_frame <= trim_start_frame or trim_end_frame >= len(qpos_array):
            raise ValueError(
                f"Invalid frame range {trim_start_frame}..{trim_end_frame} for {csv_path.name} "
                f"(available frames: 0..{len(qpos_array) - 1})"
            )
        qpos_array = qpos_array[trim_start_frame : trim_end_frame + 1].copy()

    motion_data = {
        "fps": float(fps),
        "root_pos": qpos_array[:, :3],
        "root_rot": qpos_array[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos_array[:, 7:],
        "dof_names": dof_names,
        "local_body_pos": None,
        "link_body_list": None,
        "source_file": str(csv_path),
        "source_robot": "unitree_g1",
        "target_model_file": str(target_model_path),
        "retarget_method": method,
    }
    if trim_start_frame is not None:
        motion_data["trim_start_frame"] = int(trim_start_frame)
        motion_data["trim_end_frame"] = int(trim_end_frame)

    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as file:
        pickle.dump(motion_data, file)
    convert_lens110_gmr_to_legged_lab(str(pkl_path), str(txt_path), fps=fps)
    if ground_stats is not None:
        print(
            "  grounding: "
            f"mode={ground_mode}, "
            f"foot_min/p5/mean(before)="
            f"{ground_stats['foot_min_before']:.3f}/"
            f"{ground_stats['foot_p5_before']:.3f}/"
            f"{ground_stats['foot_mean_before']:.3f}, "
            f"offset={ground_stats['ground_offset']:.3f}, "
            f"foot_ground_offset={foot_ground_offset:.3f}, "
            f"root_z_offset={root_z_offset:.3f}, "
            f"root_xy_scale={root_xy_scale:.3f}, "
            f"leg_motion_scale={leg_motion_scale:.3f}, "
            f"hip_yaw_scale={hip_yaw_scale:.3f}, "
            f"ankle_roll_scale={ankle_roll_scale:.3f}, "
            f"torso_yaw_scale={torso_yaw_scale:.3f}"
        )
    if stance_stats is not None:
        print(
            "  stance: "
            f"min_width={min_stance_width:.3f}, "
            f"frames={stance_stats['corrected_frames']}, "
            f"lat_sep min/p5/mean(before)="
            f"{stance_stats['before_min']:.3f}/"
            f"{stance_stats['before_p5']:.3f}/"
            f"{stance_stats['before_mean']:.3f}, "
            f"lat_sep min/p5/mean(after)="
            f"{stance_stats['after_min']:.3f}/"
            f"{stance_stats['after_p5']:.3f}/"
            f"{stance_stats['after_mean']:.3f}, "
            f"total_hip_roll_adjustment={stance_stats['total_hip_roll_adjustment']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=pathlib.Path,
        default=repository_root() / "tools" / "retargeting" / "robot_retargeter" / "dataset" / "lafan1_g1",
    )
    parser.add_argument(
        "--out_dir",
        type=pathlib.Path,
        default=repository_root() / "projects" / "03_walk" / "data" / "processed" / "lafan_walk_lens110_21dof",
    )
    parser.add_argument("--pattern", type=str, default="walk*.csv")
    parser.add_argument("--target_xml", type=pathlib.Path, default=lens110_mjcf())
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--method", choices=["direct", "hybrid", "ik"], default="hybrid")
    parser.add_argument("--hybrid_iterations", type=int, default=4)
    parser.add_argument("--stance_width_offset", type=float, default=0.06)
    parser.add_argument("--hip_yaw_out_offset", type=float, default=0.0)
    parser.add_argument("--override", action="store_true", default=False)
    parser.add_argument("--ground_mode", choices=["constant", "per_frame", "smooth", "off"], default="constant")
    parser.add_argument("--ground_clearance", type=float, default=0.02)
    parser.add_argument("--foot_ground_offset", type=float, default=0.0)
    parser.add_argument("--ground_percentile", type=float, default=5.0)
    parser.add_argument("--max_ground_lowering", type=float, default=0.06)
    parser.add_argument("--root_z_offset", type=float, default=0.0)
    parser.add_argument("--ground_smooth_window", type=int, default=5)
    parser.add_argument("--leg_motion_scale", type=float, default=1.0)
    parser.add_argument("--hip_yaw_scale", type=float, default=1.0)
    parser.add_argument("--ankle_roll_scale", type=float, default=1.0)
    parser.add_argument("--torso_yaw_scale", type=float, default=1.0)
    parser.add_argument("--lower_body_smooth_window", type=int, default=1)
    parser.add_argument("--min_stance_width", type=float, default=0.10)
    parser.add_argument("--stance_correction_gain", type=float, default=3.0)
    parser.add_argument("--stance_correction_iterations", type=int, default=3)
    parser.add_argument("--max_stance_correction_step", type=float, default=0.12)
    parser.add_argument("--root_xy_scale", type=float, default=1.0)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--output_suffix", type=str, default="")
    args = parser.parse_args()

    csv_files = sorted(args.src_dir.glob(args.pattern))
    pkl_dir = args.out_dir / "pkl"
    txt_dir = args.out_dir / "txt"
    print(f"Found {len(csv_files)} files")
    output_suffix = normalize_output_suffix(args.output_suffix)

    for index, csv_path in enumerate(csv_files, start=1):
        output_stem = csv_path.stem
        if args.frame_start is not None or args.frame_end is not None:
            if args.frame_start is None or args.frame_end is None:
                raise ValueError("Please provide both --frame_start and --frame_end when trimming output.")
            output_stem += f"_f{args.frame_start}_to_f{args.frame_end}"
        output_stem += output_suffix

        pkl_path = pkl_dir / f"{output_stem}.pkl"
        txt_path = txt_dir / f"{output_stem}.txt"
        if txt_path.exists() and not args.override:
            print(f"[{index}/{len(csv_files)}] skip {csv_path.name}")
            continue
        print(f"[{index}/{len(csv_files)}] retarget {csv_path.name} -> {output_stem}")
        retarget_file(
            csv_path,
            pkl_path,
            txt_path,
            args.target_xml,
            args.fps,
            args.ground_mode,
            args.ground_clearance,
            args.foot_ground_offset,
            args.ground_percentile,
            args.max_ground_lowering,
            args.root_z_offset,
            args.ground_smooth_window,
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
            args.root_xy_scale,
            args.frame_start,
            args.frame_end,
        )

    print(f"Done: {args.out_dir}")


if __name__ == "__main__":
    main()
