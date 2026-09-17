import argparse
import pathlib
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from batch_g1_csv_to_lens110 import (  # noqa: E402
    apply_lens110_motion_postprocess,
    apply_root_xy_scale,
    clip_lens110_joint_limits,
    direct_map_g1_to_lens110,
    enforce_min_stance_width,
    load_g1_csv_qpos,
    qpos_dof_names,
)
from batch_smplx_to_lens110 import ground_qpos_by_feet  # noqa: E402
from general_motion_retargeting import ROBOT_XML_DICT  # noqa: E402
from path_utils import repository_root  # noqa: E402


WORKSPACE_ROOT = repository_root()
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "projects/04_fall_to_stand/framework/amp_mjlab/AMP_mjlab/src/assets/motions/g1/amp"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "projects/04_fall_to_stand/framework/amp_mjlab/AMP_mjlab/src/assets/motions/lens110"
DEFAULT_TARGET_XML = WORKSPACE_ROOT / "frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110.xml"
DEFAULT_TENDON_SOLVER_DIR = WORKSPACE_ROOT / "projects/02_dance_half_body/framework/legged_lab_upper_lower/pitchRoll2UpperLower/solve"

if DEFAULT_TENDON_SOLVER_DIR.exists() and str(DEFAULT_TENDON_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(DEFAULT_TENDON_SOLVER_DIR))

try:
    from xml_tendon_pr_to_ul import Lens110XmlTendonSolver  # noqa: E402
except Exception:  # pragma: no cover - optional until tendon mode is requested
    Lens110XmlTendonSolver = None

LENS110_ANKLE_PITCH_ROLL_NEUTRAL = {
    "left_ankle_pitch_joint": -0.15,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.0,
}

LENS110_SIDEWAY_PARALLEL_NEUTRAL = {
    "left_hip_pitch_joint": -0.14,
    "left_hip_roll_joint": 0.01,
    "left_hip_yaw_joint": -0.10,
    "left_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.15,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.14,
    "right_hip_roll_joint": -0.01,
    "right_hip_yaw_joint": 0.10,
    "right_knee_joint": 0.30,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.0,
}

SIDEWAY_SAGITTAL_PAIRS = (
    ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("left_knee_joint", "right_knee_joint"),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
)

SIDEWAY_LATERAL_JOINTS = (
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)

SIDEWAY_YAW_JOINTS = (
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
)

SIDEWAY_LEG_JOINTS = {
    "left": (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ),
    "right": (
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ),
}

LENS110_LINKAGE_DEFAULTS = {
    "left_ankle_upper_joint": 0.161341,
    "left_ankle_lower_joint": 0.162273,
    "right_ankle_upper_joint": 0.161341,
    "right_ankle_lower_joint": 0.162273,
}

AMP_LENS110_ELBOW_JOINTS = (
    ("left_elbow_joint", -1.0),
    ("right_elbow_joint", 1.0),
)


def load_model(xml_path: pathlib.Path, mesh_dir: pathlib.Path | None = None) -> mj.MjModel:
    if mesh_dir is None:
        return mj.MjModel.from_xml_path(str(xml_path))

    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(mesh_dir))
    return mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[..., 1:] *= -1.0
    return result


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def axis_angle_from_quat(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = np.divide(quat, norm, out=np.zeros_like(quat), where=norm > 1e-12)
    xyz = quat[..., 1:]
    xyz_norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(xyz_norm, quat[..., :1])
    axis = np.divide(xyz, xyz_norm, out=np.zeros_like(xyz), where=xyz_norm > 1e-8)
    return axis * angle


def make_quat_continuous(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    flat = result.reshape(result.shape[0], -1, 4)
    for frame_id in range(1, flat.shape[0]):
        flip = np.sum(flat[frame_id - 1] * flat[frame_id], axis=-1) < 0.0
        flat[frame_id, flip] *= -1.0
    return result


def angular_velocity_from_quat(quat: np.ndarray, dt: float) -> np.ndarray:
    quat = make_quat_continuous(quat)
    if quat.shape[0] < 3:
        return np.zeros(quat.shape[:-1] + (3,), dtype=np.float32)
    q_prev = quat[:-2]
    q_next = quat[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    omega = np.concatenate((omega[:1], omega, omega[-1:]), axis=0)
    return omega.astype(np.float32)


def source_qpos_from_amp_npz(npz: np.lib.npyio.NpzFile, source_model: mj.MjModel) -> np.ndarray:
    joint_pos = np.asarray(npz["joint_pos"], dtype=np.float64)
    body_pos_w = np.asarray(npz["body_pos_w"], dtype=np.float64)
    body_quat_w = np.asarray(npz["body_quat_w"], dtype=np.float64)
    qpos = np.zeros((joint_pos.shape[0], source_model.nq), dtype=np.float64)
    qpos[:, :3] = body_pos_w[:, 0, :]
    qpos[:, 3:7] = body_quat_w[:, 0, :]
    qpos[:, 7 : 7 + joint_pos.shape[1]] = joint_pos
    return qpos


def source_qpos_and_fps(input_path: pathlib.Path, source_model: mj.MjModel, source_fps: float) -> tuple[np.ndarray, float]:
    if input_path.suffix.lower() == ".csv":
        return load_g1_csv_qpos(input_path, source_model), float(source_fps)

    with np.load(input_path) as npz:
        fps = float(np.asarray(npz["fps"]).reshape(-1)[0])
        return source_qpos_from_amp_npz(npz, source_model), fps


def fill_lens110_linkage_defaults(qpos: np.ndarray, model: mj.MjModel) -> None:
    target_names = qpos_dof_names(model)
    for joint_name, value in LENS110_LINKAGE_DEFAULTS.items():
        if joint_name in target_names:
            qpos[:, 7 + target_names.index(joint_name)] = value


def qpos_column(model: mj.MjModel, joint_name: str) -> int:
    return int(model.jnt_qposadr[model.joint(joint_name).id])


def map_ankle_pitch_roll_to_upper_lower_direct(
    qpos: np.ndarray,
    model: mj.MjModel,
    pitch_scale: float,
    roll_scale: float,
    neutralize_pitch_roll: bool,
) -> None:
    target_names = set(qpos_dof_names(model))
    required = set(LENS110_ANKLE_PITCH_ROLL_NEUTRAL) | set(LENS110_LINKAGE_DEFAULTS)
    if not required.issubset(target_names):
        missing = ", ".join(sorted(required - target_names))
        raise ValueError(f"Target model is missing Lens110 ankle joints: {missing}")

    for side in ("left", "right"):
        pitch_name = f"{side}_ankle_pitch_joint"
        roll_name = f"{side}_ankle_roll_joint"
        upper_name = f"{side}_ankle_upper_joint"
        lower_name = f"{side}_ankle_lower_joint"

        pitch = qpos[:, qpos_column(model, pitch_name)]
        roll = qpos[:, qpos_column(model, roll_name)]
        pitch_delta = pitch - LENS110_ANKLE_PITCH_ROLL_NEUTRAL[pitch_name]
        roll_delta = roll - LENS110_ANKLE_PITCH_ROLL_NEUTRAL[roll_name]

        # Direct linear approximation: common-mode tracks ankle pitch, differential
        # mode tracks ankle roll. Right side roll is mirrored by the XML geometry.
        roll_sign = 1.0 if side == "left" else -1.0
        upper = (
            LENS110_LINKAGE_DEFAULTS[upper_name]
            + pitch_scale * pitch_delta
            + roll_sign * roll_scale * roll_delta
        )
        lower = (
            LENS110_LINKAGE_DEFAULTS[lower_name]
            + pitch_scale * pitch_delta
            - roll_sign * roll_scale * roll_delta
        )
        qpos[:, qpos_column(model, upper_name)] = upper
        qpos[:, qpos_column(model, lower_name)] = lower

        if neutralize_pitch_roll:
            qpos[:, qpos_column(model, pitch_name)] = LENS110_ANKLE_PITCH_ROLL_NEUTRAL[pitch_name]
            qpos[:, qpos_column(model, roll_name)] = LENS110_ANKLE_PITCH_ROLL_NEUTRAL[roll_name]


def map_ankle_pitch_roll_to_upper_lower_tendon(
    qpos: np.ndarray,
    model: mj.MjModel,
    solver,
    fps: float,
    shrink_step: float,
) -> dict[str, float]:
    if solver is None:
        raise RuntimeError("xml_tendon_pr_to_ul.py is not importable; cannot use ankle_mode=upper_lower_tendon")

    target_names = set(qpos_dof_names(model))
    required = (
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "left_ankle_upper_joint",
        "left_ankle_lower_joint",
        "right_ankle_upper_joint",
        "right_ankle_lower_joint",
    )
    missing = [name for name in required if name not in target_names]
    if missing:
        raise ValueError(f"Target model is missing Lens110 ankle joints: {missing}")

    pr_names = required[:4]
    ul_names = required[4:]
    pr_columns = [qpos_column(model, name) for name in pr_names]
    pr_pos = np.column_stack([qpos[:, column] for column in pr_columns])
    ul_pos = np.zeros_like(pr_pos)
    neutral = np.asarray(
        [LENS110_ANKLE_PITCH_ROLL_NEUTRAL[name] for name in pr_names],
        dtype=np.float64,
    )
    last_ul: np.ndarray | None = None
    fallback_frames = 0
    min_shrink = 1.0

    for frame_id, row in enumerate(pr_pos):
        try:
            ul_pos[frame_id] = solver.solve_pr_to_ul(row[0], row[1], row[2], row[3], initial=last_ul)
            last_ul = ul_pos[frame_id]
            continue
        except RuntimeError:
            pass

        fallback_frames += 1
        solved = False
        shrink_values = np.arange(1.0 - shrink_step, -1e-9, -shrink_step, dtype=np.float64)
        for shrink in shrink_values:
            candidate = neutral + shrink * (row - neutral)
            try:
                ul_pos[frame_id] = solver.solve_pr_to_ul(
                    candidate[0],
                    candidate[1],
                    candidate[2],
                    candidate[3],
                    initial=last_ul,
                )
            except RuntimeError:
                continue
            pr_pos[frame_id] = candidate
            last_ul = ul_pos[frame_id]
            min_shrink = min(min_shrink, float(shrink))
            solved = True
            break
        if not solved:
            candidate = neutral.copy()
            ul_pos[frame_id] = solver.solve_pr_to_ul(candidate[0], candidate[1], candidate[2], candidate[3], initial=last_ul)
            pr_pos[frame_id] = candidate
            last_ul = ul_pos[frame_id]
            min_shrink = 0.0

    for index, column in enumerate(pr_columns):
        qpos[:, column] = pr_pos[:, index]
    for index, joint_name in enumerate(ul_names):
        qpos[:, qpos_column(model, joint_name)] = ul_pos[:, index]
    clip_lens110_joint_limits(qpos, model, qpos_dof_names(model))
    return {
        "ankle_ul_min": float(np.min(ul_pos)),
        "ankle_ul_max": float(np.max(ul_pos)),
        "ankle_ul_std": float(np.std(ul_pos)),
        "ankle_tendon_fallback_frames": float(fallback_frames),
        "ankle_tendon_min_shrink": float(min_shrink),
    }


def blend_joint_motion_to_neutral(
    qpos: np.ndarray,
    model: mj.MjModel,
    joint_names: tuple[str, ...],
    scale: float,
) -> None:
    target_names = set(qpos_dof_names(model))
    scale = float(scale)
    for joint_name in joint_names:
        if joint_name not in target_names:
            continue
        neutral = LENS110_SIDEWAY_PARALLEL_NEUTRAL[joint_name]
        col = qpos_column(model, joint_name)
        qpos[:, col] = neutral + scale * (qpos[:, col] - neutral)


def synchronize_sideway_sagittal_pairs(
    qpos: np.ndarray,
    model: mj.MjModel,
    blend: float,
) -> None:
    blend = float(np.clip(blend, 0.0, 1.0))
    if blend <= 0.0:
        return
    target_names = set(qpos_dof_names(model))
    for left_name, right_name in SIDEWAY_SAGITTAL_PAIRS:
        if left_name not in target_names or right_name not in target_names:
            continue
        left_col = qpos_column(model, left_name)
        right_col = qpos_column(model, right_name)
        mean = 0.5 * (qpos[:, left_col] + qpos[:, right_col])
        qpos[:, left_col] = (1.0 - blend) * qpos[:, left_col] + blend * mean
        qpos[:, right_col] = (1.0 - blend) * qpos[:, right_col] + blend * mean


def smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def sideway_swing_masks(frame_count: int, cycles: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timeline = np.linspace(0.0, max(0.1, cycles), frame_count, dtype=np.float64)
    cycle_id = np.floor(timeline).astype(np.int32)
    phase = timeline - cycle_id
    cycle_id = np.minimum(cycle_id, int(np.ceil(cycles)) - 1)

    lead_swing = smooth_pulse(phase, 0.06, 0.34)
    trail_swing = smooth_pulse(phase, 0.56, 0.84)
    step_progress = np.zeros(frame_count, dtype=np.float64)
    step_progress += 0.5 * smoothstep((phase - 0.36) / 0.14)
    step_progress += 0.5 * smoothstep((phase - 0.86) / 0.14)
    root_progress = (cycle_id + step_progress) / max(0.1, cycles)
    root_progress[-1] = 1.0
    return np.clip(root_progress, 0.0, 1.0), lead_swing, trail_swing


def smooth_pulse(phase: np.ndarray, start: float, end: float) -> np.ndarray:
    ramp = max(1e-4, (end - start) * 0.25)
    rise = smoothstep((phase - start) / ramp)
    fall = 1.0 - smoothstep((phase - (end - ramp)) / ramp)
    return np.clip(rise * fall, 0.0, 1.0)


def apply_sideway_step_timing(
    qpos: np.ndarray,
    model: mj.MjModel,
    input_path: pathlib.Path,
    args: argparse.Namespace,
) -> None:
    if not args.sideway_step_timing:
        return

    root_progress, lead_mask, trail_mask = sideway_swing_masks(len(qpos), args.sideway_step_cycles)
    root_start = qpos[0, 1]
    root_delta = qpos[-1, 1] - root_start
    qpos[:, 1] = root_start + root_delta * root_progress

    path_text = input_path.as_posix().lower()
    lead_side = "left" if "sideway_left" in path_text else "right"
    trail_side = "right" if lead_side == "left" else "left"
    side_masks = {lead_side: lead_mask, trail_side: trail_mask}
    target_names = set(qpos_dof_names(model))

    for side, joint_names in SIDEWAY_LEG_JOINTS.items():
        leg_weight = args.sideway_support_leg_motion_scale + (
            1.0 - args.sideway_support_leg_motion_scale
        ) * side_masks[side]
        for joint_name in joint_names:
            if joint_name not in target_names:
                continue
            neutral = LENS110_SIDEWAY_PARALLEL_NEUTRAL[joint_name]
            col = qpos_column(model, joint_name)
            qpos[:, col] = neutral + leg_weight * (qpos[:, col] - neutral)

    for side, swing_mask in side_masks.items():
        knee_name = f"{side}_knee_joint"
        if knee_name not in target_names:
            continue
        knee_col = qpos_column(model, knee_name)
        qpos[:, knee_col] += args.sideway_swing_knee_extra * swing_mask


def sideway_segment_progress(phase: np.ndarray, start: float, end: float) -> np.ndarray:
    return smoothstep((phase - start) / max(1e-4, end - start))


def apply_sideway_step_gait(
    qpos: np.ndarray,
    model: mj.MjModel,
    input_path: pathlib.Path,
    args: argparse.Namespace,
) -> bool:
    if not args.sideway_step_gait:
        return False

    target_names = set(qpos_dof_names(model))
    required = set(name for names in SIDEWAY_LEG_JOINTS.values() for name in names)
    if not required.issubset(target_names):
        return False

    frame_count = len(qpos)
    cycles = max(0.1, args.sideway_step_cycles)
    timeline = np.linspace(0.0, cycles, frame_count, dtype=np.float64)
    cycle_id = np.minimum(np.floor(timeline).astype(np.int32), int(np.ceil(cycles)) - 1)
    phase = timeline - cycle_id

    lead_side = "left" if "sideway_left" in input_path.as_posix().lower() else "right"
    trail_side = "right" if lead_side == "left" else "left"
    lead_sign = 1.0 if lead_side == "left" else -1.0
    trail_sign = -lead_sign

    root_progress = (
        cycle_id
        + 0.5 * sideway_segment_progress(phase, 0.42, 0.58)
        + 0.5 * sideway_segment_progress(phase, 0.86, 0.98)
    ) / cycles
    root_progress = np.clip(root_progress, 0.0, 1.0)
    root_progress[-1] = 1.0
    root_start = qpos[0, 1]
    root_delta = (qpos[-1, 1] - root_start) * args.sideway_root_lateral_scale
    qpos[:, 1] = root_start + root_delta * root_progress

    crouch = sideway_segment_progress(phase, 0.02, 0.12) * (1.0 - 0.35 * sideway_segment_progress(phase, 0.90, 1.0))
    lead_lift = smooth_pulse(phase, 0.14, 0.40)
    trail_lift = smooth_pulse(phase, 0.62, 0.84)
    lead_reach = sideway_segment_progress(phase, 0.14, 0.32) * (1.0 - sideway_segment_progress(phase, 0.48, 0.64))
    trail_reach = sideway_segment_progress(phase, 0.62, 0.76) * (1.0 - sideway_segment_progress(phase, 0.82, 0.96))

    for joint_name, neutral in LENS110_SIDEWAY_PARALLEL_NEUTRAL.items():
        if joint_name in target_names:
            qpos[:, qpos_column(model, joint_name)] = neutral

    for side in ("left", "right"):
        knee_col = qpos_column(model, f"{side}_knee_joint")
        qpos[:, knee_col] = (
            LENS110_SIDEWAY_PARALLEL_NEUTRAL[f"{side}_knee_joint"]
            + args.sideway_step_knee_bend * crouch
        )

    qpos[:, qpos_column(model, f"{lead_side}_hip_roll_joint")] += lead_sign * args.sideway_step_width * lead_reach
    qpos[:, qpos_column(model, f"{trail_side}_hip_roll_joint")] += -trail_sign * args.sideway_step_close_width * trail_reach
    qpos[:, qpos_column(model, f"{lead_side}_knee_joint")] += args.sideway_step_swing_knee * lead_lift
    qpos[:, qpos_column(model, f"{trail_side}_knee_joint")] += args.sideway_step_swing_knee * trail_lift

    lead_ankle = f"{lead_side}_ankle_roll_joint"
    trail_ankle = f"{trail_side}_ankle_roll_joint"
    qpos[:, qpos_column(model, lead_ankle)] += -lead_sign * args.sideway_step_ankle_comp * lead_reach
    qpos[:, qpos_column(model, trail_ankle)] += trail_sign * args.sideway_step_ankle_comp * trail_reach

    if args.sideway_joint_smooth_window > 1:
        columns = tuple(qpos_column(model, name) for name in required if name in target_names)
        smooth_qpos_columns(qpos, columns, args.sideway_joint_smooth_window)
    clip_lens110_joint_limits(qpos, model, qpos_dof_names(model))
    return True


def apply_sideway_knee_bend(
    qpos: np.ndarray,
    model: mj.MjModel,
    bend_offset: float,
    wave_amp: float,
    wave_cycles: float,
) -> None:
    target_names = set(qpos_dof_names(model))
    knee_names = ("left_knee_joint", "right_knee_joint")
    if not all(name in target_names for name in knee_names):
        return
    phase = np.linspace(0.0, 2.0 * np.pi * max(0.1, wave_cycles), len(qpos), dtype=np.float64)
    left_wave = 0.5 * (1.0 - np.cos(phase))
    right_wave = 0.5 * (1.0 - np.cos(phase + np.pi))
    for knee_name, wave in zip(knee_names, (left_wave, right_wave), strict=True):
        col = qpos_column(model, knee_name)
        qpos[:, col] += bend_offset + wave_amp * wave


def apply_sideway_parallel_postprocess(
    model: mj.MjModel,
    qpos: np.ndarray,
    input_path: pathlib.Path,
    args: argparse.Namespace,
) -> np.ndarray:
    if not args.sideway_parallel or "sideway" not in input_path.as_posix().lower():
        return qpos

    processed = qpos.copy()
    if apply_sideway_step_gait(processed, model, input_path, args):
        return processed

    if args.sideway_root_lateral_scale != 1.0:
        processed[:, 1] = processed[:1, 1] + args.sideway_root_lateral_scale * (processed[:, 1] - processed[:1, 1])

    sagittal_joints = tuple(name for pair in SIDEWAY_SAGITTAL_PAIRS for name in pair)
    blend_joint_motion_to_neutral(processed, model, sagittal_joints, args.sideway_sagittal_scale)
    synchronize_sideway_sagittal_pairs(processed, model, args.sideway_sagittal_pair_blend)
    blend_joint_motion_to_neutral(processed, model, SIDEWAY_LATERAL_JOINTS, args.sideway_lateral_scale)
    blend_joint_motion_to_neutral(processed, model, SIDEWAY_YAW_JOINTS, args.sideway_yaw_scale)
    apply_sideway_knee_bend(
        processed,
        model,
        args.sideway_knee_bend_offset,
        args.sideway_knee_wave_amp,
        args.sideway_knee_wave_cycles,
    )
    apply_sideway_step_timing(processed, model, input_path, args)
    if args.sideway_root_smooth_window > 1:
        root_columns = (0, 2) if args.sideway_step_timing else (0, 1, 2)
        smooth_qpos_columns(processed, root_columns, args.sideway_root_smooth_window)
    if args.sideway_joint_smooth_window > 1:
        target_names = set(qpos_dof_names(model))
        joint_names = (
            tuple(name for pair in SIDEWAY_SAGITTAL_PAIRS for name in pair)
            + SIDEWAY_LATERAL_JOINTS
            + SIDEWAY_YAW_JOINTS
        )
        columns = tuple(qpos_column(model, name) for name in joint_names if name in target_names)
        smooth_qpos_columns(processed, columns, args.sideway_joint_smooth_window)
    clip_lens110_joint_limits(processed, model, qpos_dof_names(model))
    return processed


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1 or values.shape[0] < 3:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_qpos_columns(qpos: np.ndarray, columns: tuple[int, ...], window: int) -> None:
    for col in columns:
        qpos[:, col] = smooth_series(qpos[:, col], window)


def map_amp_lens110_elbow_flexion(
    qpos: np.ndarray,
    source_qpos: np.ndarray,
    source_model: mj.MjModel,
    target_model: mj.MjModel,
    elbow_scale: float,
    elbow_symmetry: str,
    elbow_max_abs: float,
) -> None:
    """Map G1 elbow values to AMP_mjlab lens110 elbow flexion directions.

    The source G1 AMP clips contain positive and negative elbow values, but visually
    both signs represent arm-swing states where the elbow should stay flexed. The
    AMP_mjlab lens110 XML uses negative left-elbow flexion and positive right-elbow
    flexion, so preserve magnitude with abs() and apply the target-side sign.
    """
    target_names = set(qpos_dof_names(target_model))
    present = [joint_name in target_names for joint_name, _sign in AMP_LENS110_ELBOW_JOINTS]
    if elbow_symmetry == "average" and all(present):
        left_name, left_sign = AMP_LENS110_ELBOW_JOINTS[0]
        right_name, right_sign = AMP_LENS110_ELBOW_JOINTS[1]
        left_source = qpos_column(source_model, left_name)
        right_source = qpos_column(source_model, right_name)
        left_target = qpos_column(target_model, left_name)
        right_target = qpos_column(target_model, right_name)
        magnitude = elbow_scale * 0.5 * (np.abs(source_qpos[:, left_source]) + np.abs(source_qpos[:, right_source]))
        if elbow_max_abs > 0.0:
            magnitude = np.minimum(magnitude, elbow_max_abs)
        qpos[:, left_target] = left_sign * magnitude
        qpos[:, right_target] = right_sign * magnitude
        return

    for joint_name, target_sign in AMP_LENS110_ELBOW_JOINTS:
        if joint_name not in target_names:
            continue
        source_joint_id = source_model.joint(joint_name).id
        target_joint_id = target_model.joint(joint_name).id
        source_adr = source_model.jnt_qposadr[source_joint_id]
        target_adr = target_model.jnt_qposadr[target_joint_id]
        magnitude = elbow_scale * np.abs(source_qpos[:, source_adr])
        if elbow_max_abs > 0.0:
            magnitude = np.minimum(magnitude, elbow_max_abs)
        qpos[:, target_adr] = target_sign * magnitude


def adjust_shoulder_pitch(
    qpos: np.ndarray,
    model: mj.MjModel,
    target_value: float,
    blend: float,
) -> None:
    if blend <= 0.0:
        return
    blend = float(np.clip(blend, 0.0, 1.0))
    target_names = set(qpos_dof_names(model))
    for joint_name in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"):
        if joint_name not in target_names:
            continue
        col = qpos_column(model, joint_name)
        qpos[:, col] = (1.0 - blend) * qpos[:, col] + blend * target_value
    clip_lens110_joint_limits(qpos, model, qpos_dof_names(model))


def adjust_arm_swing_clearance(
    qpos: np.ndarray,
    model: mj.MjModel,
    shoulder_pitch_offset: float,
    shoulder_pitch_min: float | None,
    shoulder_pitch_max: float | None,
    shoulder_roll_out_offset: float,
) -> None:
    target_names = set(qpos_dof_names(model))

    for joint_name in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"):
        if joint_name not in target_names:
            continue
        col = qpos_column(model, joint_name)
        qpos[:, col] += shoulder_pitch_offset
        if shoulder_pitch_min is not None or shoulder_pitch_max is not None:
            qpos[:, col] = np.clip(
                qpos[:, col],
                -np.inf if shoulder_pitch_min is None else shoulder_pitch_min,
                np.inf if shoulder_pitch_max is None else shoulder_pitch_max,
            )

    roll_offsets = {
        "left_shoulder_roll_joint": shoulder_roll_out_offset,
        "right_shoulder_roll_joint": -shoulder_roll_out_offset,
    }
    for joint_name, offset in roll_offsets.items():
        if joint_name not in target_names:
            continue
        qpos[:, qpos_column(model, joint_name)] += offset
    clip_lens110_joint_limits(qpos, model, qpos_dof_names(model))


def adjust_path_elbow_flexion(
    qpos: np.ndarray,
    model: mj.MjModel,
    input_path: pathlib.Path,
    jog_scale: float,
    jog_offset_abs: float,
    jog_max_abs: float,
) -> None:
    if "jog" not in input_path.as_posix().lower():
        return
    if jog_scale == 1.0 and jog_offset_abs == 0.0 and jog_max_abs <= 0.0:
        return

    target_names = set(qpos_dof_names(model))
    for joint_name, flex_sign in AMP_LENS110_ELBOW_JOINTS:
        if joint_name not in target_names:
            continue
        col = qpos_column(model, joint_name)
        magnitude = np.abs(qpos[:, col]) * jog_scale + jog_offset_abs
        if jog_max_abs > 0.0:
            magnitude = np.minimum(magnitude, jog_max_abs)
        qpos[:, col] = flex_sign * magnitude
    clip_lens110_joint_limits(qpos, model, qpos_dof_names(model))


def forward_motion(model: mj.MjModel, qpos: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    data = mj.MjData(model)
    body_count = model.nbody - 1
    body_pos_w = np.zeros((len(qpos), body_count, 3), dtype=np.float32)
    body_quat_w = np.zeros((len(qpos), body_count, 4), dtype=np.float32)

    for frame_id, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        mj.mj_forward(model, data)
        body_pos_w[frame_id] = data.xpos[1:].astype(np.float32)
        body_quat_w[frame_id] = data.xquat[1:].astype(np.float32)

    dt = 1.0 / fps
    joint_pos = qpos[:, 7:].astype(np.float32)
    joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)
    body_ang_vel_w = angular_velocity_from_quat(body_quat_w, dt)

    return {
        "fps": np.asarray([fps], dtype=np.float64),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_names": np.asarray(qpos_dof_names(model)),
        "body_names": np.asarray(
            [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
        ),
    }


def convert_file(
    source_model: mj.MjModel,
    target_model: mj.MjModel,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    args: argparse.Namespace,
) -> dict[str, float]:
    source_qpos, fps = source_qpos_and_fps(input_path, source_model, args.source_fps)

    qpos = direct_map_g1_to_lens110(
        source_model,
        target_model,
        source_qpos,
        args.stance_width_offset,
        args.hip_yaw_out_offset,
    )
    map_amp_lens110_elbow_flexion(
        qpos,
        source_qpos,
        source_model,
        target_model,
        args.elbow_scale,
        args.elbow_symmetry,
        args.elbow_max_abs,
    )
    adjust_shoulder_pitch(qpos, target_model, args.shoulder_pitch_target, args.shoulder_pitch_blend)
    adjust_arm_swing_clearance(
        qpos,
        target_model,
        args.shoulder_pitch_offset,
        args.shoulder_pitch_min,
        args.shoulder_pitch_max,
        args.shoulder_roll_out_offset,
    )
    adjust_path_elbow_flexion(
        qpos,
        target_model,
        input_path,
        args.jog_elbow_scale,
        args.jog_elbow_offset_abs,
        args.jog_elbow_max_abs,
    )
    fill_lens110_linkage_defaults(qpos, target_model)
    clip_lens110_joint_limits(qpos, target_model, qpos_dof_names(target_model))
    qpos = apply_lens110_motion_postprocess(
        target_model,
        qpos,
        args.leg_motion_scale,
        args.hip_yaw_scale,
        args.ankle_roll_scale,
        args.torso_yaw_scale,
        args.lower_body_smooth_window,
    )
    qpos = apply_sideway_parallel_postprocess(target_model, qpos, input_path, args)
    if args.ankle_mode == "pitch_roll":
        fill_lens110_linkage_defaults(qpos, target_model)
    elif args.ankle_mode == "upper_lower_direct":
        map_ankle_pitch_roll_to_upper_lower_direct(
            qpos,
            target_model,
            args.direct_ankle_pitch_scale,
            args.direct_ankle_roll_scale,
            args.direct_ankle_neutralize_pitch_roll,
        )
        clip_lens110_joint_limits(qpos, target_model, qpos_dof_names(target_model))
    elif args.ankle_mode == "upper_lower_tendon":
        ankle_stats = map_ankle_pitch_roll_to_upper_lower_tendon(
            qpos,
            target_model,
            args.tendon_solver,
            fps,
            args.tendon_shrink_step,
        )
    else:
        raise ValueError(f"Unsupported ankle_mode: {args.ankle_mode}")
    qpos, stance_stats = enforce_min_stance_width(
        target_model,
        qpos,
        args.min_stance_width,
        args.stance_correction_gain,
        args.stance_correction_iterations,
        args.max_stance_correction_step,
    )
    if args.ground_mode != "off":
        qpos, ground_stats = ground_qpos_by_feet(
            target_model,
            qpos,
            ("left_ankle_roll_link", "right_ankle_roll_link"),
            args.ground_clearance,
            args.foot_ground_offset,
            args.ground_mode,
            args.ground_percentile,
            args.max_ground_lowering,
            args.ground_smooth_window,
        )
    else:
        ground_stats = None
    if args.root_z_offset != 0.0:
        qpos[:, 2] += args.root_z_offset
    qpos = apply_root_xy_scale(qpos, args.root_xy_scale)

    output = forward_motion(target_model, qpos, fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **output)

    stats = {
        "frames": float(len(qpos)),
        "fps": fps,
        "joint_dim": float(output["joint_pos"].shape[1]),
        "body_dim": float(output["body_pos_w"].shape[1]),
    }
    if ground_stats is not None:
        stats["ground_offset"] = float(ground_stats["ground_offset"])
    if stance_stats is not None:
        stats["stance_corrected_frames"] = float(stance_stats["corrected_frames"])
    if args.ankle_mode == "upper_lower_tendon":
        stats.update(ankle_stats)
    return stats


def iter_input_files(input_dir: pathlib.Path, pattern: str) -> list[pathlib.Path]:
    return sorted(path for path in input_dir.rglob(pattern) if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=pathlib.Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target_xml", type=pathlib.Path, default=DEFAULT_TARGET_XML)
    parser.add_argument("--target_mesh_dir", type=pathlib.Path, default=None)
    parser.add_argument("--pattern", type=str, default="*.npz")
    parser.add_argument("--source_fps", type=float, default=120.0)
    parser.add_argument("--override", action="store_true", default=False)
    parser.add_argument("--stance_width_offset", type=float, default=0.06)
    parser.add_argument("--hip_yaw_out_offset", type=float, default=0.0)
    parser.add_argument("--ground_mode", choices=["constant", "per_frame", "smooth", "off"], default="constant")
    parser.add_argument("--ground_clearance", type=float, default=0.0)
    parser.add_argument("--foot_ground_offset", type=float, default=0.0)
    parser.add_argument("--ground_percentile", type=float, default=5.0)
    parser.add_argument("--max_ground_lowering", type=float, default=0.2)
    parser.add_argument("--root_z_offset", type=float, default=0.0)
    parser.add_argument("--ground_smooth_window", type=int, default=5)
    parser.add_argument("--leg_motion_scale", type=float, default=1.0)
    parser.add_argument("--hip_yaw_scale", type=float, default=1.0)
    parser.add_argument("--ankle_roll_scale", type=float, default=1.0)
    parser.add_argument("--torso_yaw_scale", type=float, default=1.0)
    parser.add_argument("--lower_body_smooth_window", type=int, default=1)
    parser.add_argument("--elbow_scale", type=float, default=1.0)
    parser.add_argument("--elbow_symmetry", choices=["none", "average"], default="average")
    parser.add_argument("--elbow_max_abs", type=float, default=0.0)
    parser.add_argument("--shoulder_pitch_target", type=float, default=0.0)
    parser.add_argument("--shoulder_pitch_blend", type=float, default=0.0)
    parser.add_argument("--shoulder_pitch_offset", type=float, default=0.0)
    parser.add_argument("--shoulder_pitch_min", type=float, default=None)
    parser.add_argument("--shoulder_pitch_max", type=float, default=None)
    parser.add_argument("--shoulder_roll_out_offset", type=float, default=0.0)
    parser.add_argument("--jog_elbow_scale", type=float, default=1.0)
    parser.add_argument("--jog_elbow_offset_abs", type=float, default=0.0)
    parser.add_argument("--jog_elbow_max_abs", type=float, default=0.0)
    parser.add_argument("--ankle_mode", choices=["pitch_roll", "upper_lower_direct", "upper_lower_tendon"], default="pitch_roll")
    parser.add_argument("--direct_ankle_pitch_scale", type=float, default=1.0)
    parser.add_argument("--direct_ankle_roll_scale", type=float, default=1.0)
    parser.add_argument("--direct_ankle_neutralize_pitch_roll", action="store_true", default=False)
    parser.add_argument("--tendon_xml", type=pathlib.Path, default=None)
    parser.add_argument("--tendon_grid_samples", type=int, default=801)
    parser.add_argument("--tendon_length_tol", type=float, default=1e-8)
    parser.add_argument("--tendon_shrink_step", type=float, default=0.05)
    parser.add_argument("--sideway_parallel", action="store_true", default=False)
    parser.add_argument("--sideway_root_lateral_scale", type=float, default=0.50)
    parser.add_argument("--sideway_sagittal_scale", type=float, default=0.35)
    parser.add_argument("--sideway_sagittal_pair_blend", type=float, default=0.35)
    parser.add_argument("--sideway_lateral_scale", type=float, default=0.45)
    parser.add_argument("--sideway_yaw_scale", type=float, default=0.10)
    parser.add_argument("--sideway_root_smooth_window", type=int, default=9)
    parser.add_argument("--sideway_joint_smooth_window", type=int, default=9)
    parser.add_argument("--sideway_knee_bend_offset", type=float, default=0.0)
    parser.add_argument("--sideway_knee_wave_amp", type=float, default=0.0)
    parser.add_argument("--sideway_knee_wave_cycles", type=float, default=3.0)
    parser.add_argument("--sideway_step_timing", action="store_true", default=False)
    parser.add_argument("--sideway_step_cycles", type=float, default=3.0)
    parser.add_argument("--sideway_support_leg_motion_scale", type=float, default=0.04)
    parser.add_argument("--sideway_swing_knee_extra", type=float, default=0.08)
    parser.add_argument("--sideway_step_gait", action="store_true", default=False)
    parser.add_argument("--sideway_step_width", type=float, default=0.22)
    parser.add_argument("--sideway_step_close_width", type=float, default=0.18)
    parser.add_argument("--sideway_step_knee_bend", type=float, default=0.18)
    parser.add_argument("--sideway_step_swing_knee", type=float, default=0.12)
    parser.add_argument("--sideway_step_ankle_comp", type=float, default=0.06)
    parser.add_argument("--min_stance_width", type=float, default=0.10)
    parser.add_argument("--stance_correction_gain", type=float, default=3.0)
    parser.add_argument("--stance_correction_iterations", type=int, default=3)
    parser.add_argument("--max_stance_correction_step", type=float, default=0.12)
    parser.add_argument("--root_xy_scale", type=float, default=1.0)
    args = parser.parse_args()

    target_mesh_dir = args.target_mesh_dir
    if target_mesh_dir is None:
        target_mesh_dir = args.target_xml.parent.parent / "meshes"

    source_model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
    target_model = load_model(args.target_xml, target_mesh_dir)
    args.tendon_solver = None
    if args.ankle_mode == "upper_lower_tendon":
        if Lens110XmlTendonSolver is None:
            raise RuntimeError(f"Could not import Lens110XmlTendonSolver from {DEFAULT_TENDON_SOLVER_DIR}")
        tendon_xml = args.target_xml if args.tendon_xml is None else args.tendon_xml
        args.tendon_solver = Lens110XmlTendonSolver(
            tendon_xml,
            grid_samples=args.tendon_grid_samples,
            length_tol=args.tendon_length_tol,
        )
    input_files = iter_input_files(args.input_dir, args.pattern)
    if not input_files:
        raise FileNotFoundError(f"No files matched {args.pattern!r} under {args.input_dir}")

    print(f"Source: {args.input_dir}")
    print(f"Target XML: {args.target_xml}")
    print(f"Output: {args.output_dir}")
    print(f"Ankle mode: {args.ankle_mode}")
    print(f"Sideway parallel: {args.sideway_parallel}")
    print(f"Files: {len(input_files)}")

    for index, input_path in enumerate(input_files, start=1):
        rel_path = input_path.relative_to(args.input_dir)
        output_path = (args.output_dir / rel_path).with_suffix(".npz")
        if output_path.exists() and not args.override:
            print(f"[{index}/{len(input_files)}] skip {rel_path}")
            continue
        stats = convert_file(source_model, target_model, input_path, output_path, args)
        print(
            f"[{index}/{len(input_files)}] wrote {rel_path} "
            f"frames={int(stats['frames'])} joints={int(stats['joint_dim'])} bodies={int(stats['body_dim'])}"
        )


if __name__ == "__main__":
    main()
