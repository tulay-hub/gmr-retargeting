import argparse
import pathlib
import pickle
import sys

import mujoco as mj
import numpy as np

from general_motion_retargeting import ROBOT_XML_DICT

HERE = pathlib.Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from batch_smplx_to_lens110 import ground_qpos_by_feet
from lens110_gmr_to_legged_lab import convert_lens110_gmr_to_legged_lab


DEFAULT_ROOT_POS = np.array([0.0, 0.0, 0.68], dtype=np.float64)
DEFAULT_ROOT_ROT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.14,
    "left_hip_roll_joint": 0.01,
    "left_hip_yaw_joint": -0.1,
    "left_knee_joint": 0.36,
    "left_ankle_pitch_joint": -0.20,
    "left_ankle_roll_joint": -0.20,
    "right_hip_pitch_joint": -0.14,
    "right_hip_roll_joint": -0.01,
    "right_hip_yaw_joint": 0.1,
    "right_knee_joint": 0.36,
    "right_ankle_pitch_joint": -0.20,
    "right_ankle_roll_joint": -0.20,
    "torso_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.4,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.8,
    "right_shoulder_pitch_joint": 0.4,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.8,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=pathlib.Path, required=True)
    parser.add_argument("--output_pkl", type=pathlib.Path, default=None)
    parser.add_argument("--start_frame", type=int, required=True)
    parser.add_argument("--end_frame", type=int, required=True)
    parser.add_argument("--prepend_init_frames", type=int, default=30)
    parser.add_argument("--append_init_frames", type=int, default=30)
    parser.add_argument("--ground_mode", choices=["constant", "per_frame", "smooth", "off"], default="smooth")
    parser.add_argument("--ground_clearance", type=float, default=0.02)
    parser.add_argument("--ground_percentile", type=float, default=5.0)
    parser.add_argument("--max_ground_lowering", type=float, default=0.06)
    parser.add_argument("--ground_smooth_window", type=int, default=31)
    return parser.parse_args()


def infer_output_pkl(input_pkl, start_frame, end_frame):
    return input_pkl.with_name(f"{input_pkl.stem}_f{start_frame}_to_f{end_frame}_initblend.pkl")


def infer_output_txt(output_pkl):
    if output_pkl.parent.name == "pkl":
        dataset_dir = output_pkl.parent.parent
        return dataset_dir / "txt" / f"{output_pkl.stem}.txt"
    return output_pkl.with_suffix(".txt")


def load_motion_data(path):
    with path.open("rb") as file:
        return pickle.load(file)


def build_default_dof(dof_names):
    values = []
    for name in dof_names:
        if name not in DEFAULT_JOINT_POS:
            raise KeyError(f"Missing default init_state joint value for {name}")
        values.append(DEFAULT_JOINT_POS[name])
    return np.asarray(values, dtype=np.float64)


def normalize_quat_xyzw(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return DEFAULT_ROOT_ROT_XYZW.copy()
    return quat / norm


def slerp_xyzw(q0, q1, alpha):
    q0 = normalize_quat_xyzw(q0)
    q1 = normalize_quat_xyzw(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        return normalize_quat_xyzw((1.0 - alpha) * q0 + alpha * q1)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def build_transition_frames(start_value, end_value, num_frames, include_start, include_end, interp_fn):
    if num_frames <= 0:
        shape = (0,) + np.asarray(start_value).shape
        return np.empty(shape, dtype=np.float64)
    alphas = np.linspace(0.0, 1.0, num_frames, endpoint=include_end, dtype=np.float64)
    if not include_start:
        alphas = np.linspace(0.0, 1.0, num_frames + 1, endpoint=include_end, dtype=np.float64)[1:]
    return np.asarray([interp_fn(start_value, end_value, alpha) for alpha in alphas], dtype=np.float64)


def linear_interp(start_value, end_value, alpha):
    return (1.0 - alpha) * np.asarray(start_value, dtype=np.float64) + alpha * np.asarray(end_value, dtype=np.float64)


def trim_and_blend_motion(data, start_frame, end_frame, prepend_init_frames, append_init_frames):
    total_frames = len(data["root_pos"])
    if start_frame < 0 or end_frame < start_frame or end_frame >= total_frames:
        raise ValueError(f"Invalid frame range {start_frame}..{end_frame} for motion with {total_frames} frames")

    stop_frame = end_frame + 1
    root_pos = np.asarray(data["root_pos"][start_frame:stop_frame], dtype=np.float64).copy()
    root_rot = np.asarray(data["root_rot"][start_frame:stop_frame], dtype=np.float64).copy()
    dof_pos = np.asarray(data["dof_pos"][start_frame:stop_frame], dtype=np.float64).copy()
    dof_names = data["dof_names"]

    default_dof = build_default_dof(dof_names)

    prepend_root_pos = build_transition_frames(
        DEFAULT_ROOT_POS,
        root_pos[0],
        prepend_init_frames,
        include_start=True,
        include_end=False,
        interp_fn=linear_interp,
    )
    prepend_root_rot = build_transition_frames(
        DEFAULT_ROOT_ROT_XYZW,
        root_rot[0],
        prepend_init_frames,
        include_start=True,
        include_end=False,
        interp_fn=slerp_xyzw,
    )
    prepend_dof_pos = build_transition_frames(
        default_dof,
        dof_pos[0],
        prepend_init_frames,
        include_start=True,
        include_end=False,
        interp_fn=linear_interp,
    )

    append_root_pos = build_transition_frames(
        root_pos[-1],
        DEFAULT_ROOT_POS,
        append_init_frames,
        include_start=False,
        include_end=True,
        interp_fn=linear_interp,
    )
    append_root_rot = build_transition_frames(
        root_rot[-1],
        DEFAULT_ROOT_ROT_XYZW,
        append_init_frames,
        include_start=False,
        include_end=True,
        interp_fn=slerp_xyzw,
    )
    append_dof_pos = build_transition_frames(
        dof_pos[-1],
        default_dof,
        append_init_frames,
        include_start=False,
        include_end=True,
        interp_fn=linear_interp,
    )

    blended = dict(data)
    blended["root_pos"] = np.concatenate([prepend_root_pos, root_pos, append_root_pos], axis=0)
    blended["root_rot"] = np.concatenate([prepend_root_rot, root_rot, append_root_rot], axis=0)
    blended["dof_pos"] = np.concatenate([prepend_dof_pos, dof_pos, append_dof_pos], axis=0)
    blended["trim_source_file"] = str(data.get("trim_source_file", data.get("source_file", "")))
    blended["trim_start_frame"] = int(start_frame)
    blended["trim_end_frame"] = int(end_frame)
    blended["prepend_init_frames"] = int(prepend_init_frames)
    blended["append_init_frames"] = int(append_init_frames)
    return blended


def motion_to_qpos_array(motion_data):
    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(motion_data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float64)
    qpos = np.zeros((len(root_pos), 7 + dof_pos.shape[1]), dtype=np.float64)
    qpos[:, :3] = root_pos
    qpos[:, 3:7] = root_rot_xyzw[:, [3, 0, 1, 2]]
    qpos[:, 7:] = dof_pos
    return qpos


def qpos_array_to_motion(motion_data, qpos_array):
    updated = dict(motion_data)
    updated["root_pos"] = qpos_array[:, :3].copy()
    updated["root_rot"] = qpos_array[:, 3:7][:, [1, 2, 3, 0]].copy()
    updated["dof_pos"] = qpos_array[:, 7:].copy()
    return updated


def ground_motion(motion_data, ground_mode, ground_clearance, ground_percentile, max_ground_lowering, ground_smooth_window):
    if ground_mode == "off":
        return motion_data, None
    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["lens110_21dof"]))
    qpos_array = motion_to_qpos_array(motion_data)
    qpos_array, ground_stats = ground_qpos_by_feet(
        model,
        qpos_array,
        ("left_ankle_roll_link", "right_ankle_roll_link"),
        ground_clearance,
        ground_mode,
        ground_percentile,
        max_ground_lowering,
        ground_smooth_window,
    )
    return qpos_array_to_motion(motion_data, qpos_array), ground_stats


def save_motion_data(path, motion_data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(motion_data, file)


def main():
    args = parse_args()
    output_pkl = args.output_pkl or infer_output_pkl(args.input_pkl, args.start_frame, args.end_frame)
    output_txt = infer_output_txt(output_pkl)

    motion_data = load_motion_data(args.input_pkl)
    motion_data = trim_and_blend_motion(
        motion_data,
        args.start_frame,
        args.end_frame,
        args.prepend_init_frames,
        args.append_init_frames,
    )
    motion_data, ground_stats = ground_motion(
        motion_data,
        args.ground_mode,
        args.ground_clearance,
        args.ground_percentile,
        args.max_ground_lowering,
        args.ground_smooth_window,
    )
    motion_data["retarget_method"] = f"{motion_data.get('retarget_method', 'unknown')}_trim_blend"
    motion_data["ground_mode"] = args.ground_mode

    save_motion_data(output_pkl, motion_data)
    convert_lens110_gmr_to_legged_lab(str(output_pkl), str(output_txt), fps=motion_data.get("fps"))

    total_frames = len(motion_data["root_pos"])
    print(f"Saved PKL: {output_pkl}")
    print(f"Saved TXT: {output_txt}")
    print(
        f"Frames: {total_frames} "
        f"(trim={args.start_frame}->{args.end_frame}, "
        f"prepend={args.prepend_init_frames}, append={args.append_init_frames})"
    )
    if ground_stats is not None:
        print(
            "Grounding: "
            f"mode={args.ground_mode}, "
            f"foot_min/p5/mean(before)="
            f"{ground_stats['foot_min_before']:.3f}/"
            f"{ground_stats['foot_p5_before']:.3f}/"
            f"{ground_stats['foot_mean_before']:.3f}, "
            f"offset={ground_stats['ground_offset']:.3f}"
        )


if __name__ == "__main__":
    main()
