"""Booster K1 CSV -> lens110_21dof 动作转换。

输入格式是 K1 qpos CSV: root_pos(3) + root_rot xyzw(4) + 22 关节。
输出保持 lens110 部署格式: root_pos(3) + root_rot xyzw(4) + 21 关节。
K1 没有腰部 yaw 和独立肩 yaw，先用最近的关节做初值，再由 hybrid IK 修正手臂。
"""

import argparse
import pathlib
import pickle
import sys

import mujoco as mj
import mink
import numpy as np
from scipy.optimize import minimize

HERE = pathlib.Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from batch_g1_csv_to_lens110 import (
    clip_lens110_joint_limits,
    apply_lens110_motion_postprocess,
    enforce_min_stance_width,
    get_dof_indices_by_joint_names,
    qpos_dof_names,
)
from batch_smplx_to_lens110 import ground_qpos_by_feet
from lens110_gmr_to_legged_lab import convert_lens110_gmr_to_legged_lab
from lens110_pkl_to_training import (
    TRAIN_JOINT_ORDER,
    report_joint_limits,
    resample_poses,
    reorder_dof,
    write_csv_28,
)
from path_utils import lens110_mjcf, repository_root  # noqa: E402

ROOT = repository_root()


DEFAULT_TARGET_XML = (
    lens110_mjcf()
)

# K1 关节名没有 _joint 后缀；roll/肘部方向与 lens110 的装配方向不同。
K1_JOINT_MAP = {
    "left_hip_pitch_joint": ("Left_Hip_Pitch", 1.0),
    "left_hip_roll_joint": ("Left_Hip_Roll", 1.0),
    "left_hip_yaw_joint": ("Left_Hip_Yaw", 1.0),
    "left_knee_joint": ("Left_Knee_Pitch", 1.0),
    "left_ankle_pitch_joint": ("Left_Ankle_Pitch", 1.0),
    "left_ankle_roll_joint": ("Left_Ankle_Roll", 1.0),
    "right_hip_pitch_joint": ("Right_Hip_Pitch", 1.0),
    "right_hip_roll_joint": ("Right_Hip_Roll", 1.0),
    "right_hip_yaw_joint": ("Right_Hip_Yaw", 1.0),
    "right_knee_joint": ("Right_Knee_Pitch", 1.0),
    "right_ankle_pitch_joint": ("Right_Ankle_Pitch", 1.0),
    "right_ankle_roll_joint": ("Right_Ankle_Roll", 1.0),
    "left_shoulder_pitch_joint": ("ALeft_Shoulder_Pitch", 1.0),
    "left_shoulder_roll_joint": ("Left_Shoulder_Roll", -1.0),
    "left_shoulder_yaw_joint": ("Left_Elbow_Yaw", 1.0),
    "left_elbow_joint": ("Left_Elbow_Pitch", -1.0),
    "right_shoulder_pitch_joint": ("ARight_Shoulder_Pitch", 1.0),
    "right_shoulder_roll_joint": ("Right_Shoulder_Roll", 1.0),
    "right_shoulder_yaw_joint": ("Right_Elbow_Yaw", 1.0),
    "right_elbow_joint": ("Right_Elbow_Pitch", -1.0),
}

K1_HYBRID_TASK_PAIRS = (
    ("left_elbow_link", "Left_Arm_3", 80.0),
    ("right_elbow_link", "Right_Arm_3", 80.0),
    ("left_hand", "left_hand_link", 160.0),
    ("right_hand", "right_hand_link", 160.0),
)


def load_k1_csv_qpos(csv_path, model):
    values = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.shape[1] != model.nq:
        raise ValueError(
            f"{csv_path} has {values.shape[1]} columns, expected K1 nq={model.nq}"
        )
    qpos = values.copy()
    # CSV 是 xyzw，MuJoCo free joint 是 wxyz。
    qpos[:, 3:7] = values[:, 3:7][:, [3, 0, 1, 2]]
    return qpos


def direct_map_k1_to_lens110(
    source_model,
    target_model,
    source_qpos,
    stance_width_offset,
    hip_yaw_out_offset,
):
    source_names = qpos_dof_names(source_model)
    target_names = qpos_dof_names(target_model)
    source_index = {name: 7 + index for index, name in enumerate(source_names)}

    target_qpos = np.zeros((len(source_qpos), target_model.nq), dtype=np.float64)
    target_qpos[:, :7] = source_qpos[:, :7]
    for target_index, target_name in enumerate(target_names):
        item = K1_JOINT_MAP.get(target_name)
        if item is None:
            continue
        source_name, sign = item
        target_qpos[:, 7 + target_index] = (
            sign * source_qpos[:, source_index[source_name]]
        )

    from batch_g1_csv_to_lens110 import apply_lens110_leg_adjustments

    apply_lens110_leg_adjustments(
        target_qpos,
        target_names,
        stance_width_offset,
        hip_yaw_out_offset,
    )
    for target_index, target_name in enumerate(target_names):
        joint_id = target_model.joint(target_name).id
        if target_model.jnt_limited[joint_id]:
            lower, upper = target_model.jnt_range[joint_id]
            target_qpos[:, 7 + target_index] = np.clip(
                target_qpos[:, 7 + target_index], lower, upper
            )
    return target_qpos


def hybrid_map_k1_to_lens110(
    source_model,
    target_model,
    source_qpos,
    hybrid_iterations,
    arm_scale,
    stance_width_offset,
    hip_yaw_out_offset,
):
    qpos_array = direct_map_k1_to_lens110(
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
    arm_dof_indices = set(
        get_dof_indices_by_joint_names(target_model, arm_joint_names)
    )
    frozen_dof_indices = [
        index for index in range(target_model.nv) if index not in arm_dof_indices
    ]
    freeze_non_arm = mink.DofFreezingTask(target_model, frozen_dof_indices)
    limits = [mink.ConfigurationLimit(target_model)]
    tasks = [
        mink.FrameTask(
            frame_name=target_body,
            frame_type="body",
            position_cost=position_cost,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        for target_body, _, position_cost in K1_HYBRID_TASK_PAIRS
    ]
    source_body_ids = {
        source_body: source_model.body(source_body).id
        for _, source_body, _ in K1_HYBRID_TASK_PAIRS
    }
    source_root_id = source_model.body("Trunk").id
    dt = target_model.opt.timestep

    for frame_index, source_frame_qpos in enumerate(source_qpos):
        source_data.qpos[:] = source_frame_qpos
        mj.mj_forward(source_model, source_data)
        root_world = source_data.xpos[source_root_id].copy()
        configuration.update(qpos_array[frame_index])
        for task, (target_body, source_body, _) in zip(tasks, K1_HYBRID_TASK_PAIRS):
            source_body_id = source_body_ids[source_body]
            relative = source_data.xpos[source_body_id] - root_world
            task.set_target(
                mink.SE3.from_rotation_and_translation(
                    mink.SO3(source_data.xquat[source_body_id].copy()),
                    root_world + arm_scale * relative,
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


def apply_left_shoulder_roll_offset(target_model, qpos_array, offset):
    if offset == 0.0:
        return
    joint_id = target_model.joint("left_shoulder_roll_joint").id
    qpos_adr = target_model.jnt_qposadr[joint_id]
    lower, upper = target_model.jnt_range[joint_id]
    qpos_array[:, qpos_adr] = np.clip(qpos_array[:, qpos_adr] + offset, lower, upper)


def apply_right_shoulder_roll_offset(target_model, qpos_array, offset):
    if offset == 0.0:
        return
    joint_id = target_model.joint("right_shoulder_roll_joint").id
    qpos_adr = target_model.jnt_qposadr[joint_id]
    lower, upper = target_model.jnt_range[joint_id]
    qpos_array[:, qpos_adr] = np.clip(qpos_array[:, qpos_adr] + offset, lower, upper)


def limit_joint_speeds(target_model, qpos_array, fps, max_velocity):
    if max_velocity <= 0.0 or fps <= 0.0:
        return
    max_step = float(max_velocity) / float(fps)
    joint_qpos_adrs = [
        target_model.jnt_qposadr[joint_id]
        for joint_id in range(target_model.njnt)
        if target_model.jnt_type[joint_id] != mj.mjtJoint.mjJNT_FREE
    ]
    for qpos_adr in joint_qpos_adrs:
        joint_id = int(np.flatnonzero(target_model.jnt_qposadr == qpos_adr)[0])
        if target_model.jnt_limited[joint_id]:
            lower, upper = target_model.jnt_range[joint_id]
        else:
            lower, upper = -10.0, 10.0
        column = qpos_array[:, qpos_adr]
        for frame_id in range(1, len(column)):
            column[frame_id] = np.clip(
                column[frame_id], column[frame_id - 1] - max_step, column[frame_id - 1] + max_step
            )
        for frame_id in range(len(column) - 2, -1, -1):
            column[frame_id] = np.clip(
                column[frame_id], column[frame_id + 1] - max_step, column[frame_id + 1] + max_step
            )
        qpos_array[:, qpos_adr] = np.clip(column, lower, upper)


def build_nonadjacent_geom_pairs(model):
    """取非相邻 body 的 geom 对，用于自碰审计；地面单独加入。"""
    pairs = []
    for body_id_1 in range(1, model.nbody):
        for body_id_2 in range(body_id_1 + 1, model.nbody):
            if (
                body_id_1 == model.body_parentid[body_id_2]
                or body_id_2 == model.body_parentid[body_id_1]
            ):
                continue
            geoms_1 = [
                geom_id
                for geom_id in range(model.ngeom)
                if model.geom_bodyid[geom_id] == body_id_1
            ]
            geoms_2 = [
                geom_id
                for geom_id in range(model.ngeom)
                if model.geom_bodyid[geom_id] == body_id_2
            ]
            for geom_id_1 in geoms_1:
                for geom_id_2 in geoms_2:
                    pairs.append((geom_id_1, geom_id_2))
    world_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] == 0
    ]
    robot_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] > 0
    ]
    for geom_id_1 in world_geoms:
        for geom_id_2 in robot_geoms:
            pairs.append((geom_id_1, geom_id_2))
    return pairs


def resolve_self_collisions(
    model,
    qpos_array,
    pairs,
    clearance=0.0015,
    max_iterations=8,
    smooth_window=5,
):
    """对穿透帧做最小幅值关节修正，随后平滑修正量。"""
    data = mj.MjData(model)
    fromto = np.zeros(6)
    base_qpos = qpos_array.copy()
    joint_qpos_adrs = [
        model.jnt_qposadr[joint_id]
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mj.mjtJoint.mjJNT_FREE
    ]
    bounds = []
    for qpos_adr in joint_qpos_adrs:
        joint_id = int(np.flatnonzero(model.jnt_qposadr == qpos_adr)[0])
        if model.jnt_limited[joint_id]:
            bounds.append(tuple(model.jnt_range[joint_id]))
        else:
            bounds.append((-10.0, 10.0))

    def min_distance(qpos, return_penetrations=False):
        data.qpos[:3] = qpos[:3]
        data.qpos[3:7] = qpos[3:7]
        data.qpos[7:] = qpos[7:]
        mj.mj_forward(model, data)
        minimum = 1e9
        penetrations = {}
        centers = data.geom_xpos
        radii = model.geom_rbound
        for geom_id_1, geom_id_2 in pairs:
            offset = centers[geom_id_2] - centers[geom_id_1]
            broad_radius = radii[geom_id_1] + radii[geom_id_2] + 0.05
            if np.dot(offset, offset) > broad_radius * broad_radius:
                continue
            distance = mj.mj_geomDistance(
                model, data, geom_id_1, geom_id_2, 0.02, fromto
            )
            minimum = min(minimum, distance)
            if distance < clearance:
                key = (geom_id_1, geom_id_2)
                penetrations[key] = min(penetrations.get(key, 1e9), distance)
        return (minimum, penetrations) if return_penetrations else minimum

    def objective(z, reference, root_pos, root_quat):
        loss = float(np.sum(((z - reference) / 0.04) ** 2))
        data.qpos[:3] = root_pos
        data.qpos[3:7] = root_quat
        data.qpos[7:] = z
        mj.mj_forward(model, data)
        centers = data.geom_xpos
        radii = model.geom_rbound
        for geom_id_1, geom_id_2 in pairs:
            offset = centers[geom_id_2] - centers[geom_id_1]
            broad_radius = radii[geom_id_1] + radii[geom_id_2] + 0.05
            if np.dot(offset, offset) > broad_radius * broad_radius:
                continue
            distance = mj.mj_geomDistance(
                model, data, geom_id_1, geom_id_2, 0.02, fromto
            )
            if distance < clearance:
                loss += 8.0 * ((clearance - distance) / 0.005) ** 2
        return loss

    deltas = np.zeros((len(qpos_array), len(joint_qpos_adrs)), dtype=np.float64)
    resolved_iterations = 0
    for _ in range(max_iterations):
        trial = base_qpos.copy()
        trial[:, 7:] += deltas
        for dof_index, (lower, upper) in enumerate(bounds):
            trial[:, 7 + dof_index] = np.clip(
                trial[:, 7 + dof_index], lower, upper
            )
        bad_frames = [
            frame_id
            for frame_id in range(len(trial))
            if min_distance(trial[frame_id]) < 0.0
        ]
        if not bad_frames:
            break

        for frame_id in bad_frames:
            trial_qpos = trial[frame_id]
            initial = trial_qpos[7:] - deltas[frame_id]
            result = minimize(
                objective,
                initial + deltas[frame_id],
                args=(initial, trial_qpos[:3], trial_qpos[3:7]),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": 35,
                    "ftol": 1e-9,
                    "maxls": 20,
                    "eps": 0.003,
                },
            )
            deltas[frame_id] = result.x - initial

        window = max(1, int(smooth_window))
        if window > 1:
            if window % 2 == 0:
                window += 1
            pad = window // 2
            kernel = np.ones(window, dtype=np.float64) / window
            smoothed = np.empty_like(deltas)
            for dof_index in range(deltas.shape[1]):
                padded = np.pad(
                    deltas[:, dof_index], (pad, pad), mode="edge"
                )
                smoothed[:, dof_index] = np.convolve(padded, kernel, mode="valid")
            deltas = smoothed
        resolved_iterations += 1

    # 平滑可能把极小残余带回穿透；最终阶段只处理剩余帧，不再回填平均。
    for _ in range(3):
        trial = base_qpos.copy()
        trial[:, 7:] += deltas
        for dof_index, (lower, upper) in enumerate(bounds):
            trial[:, 7 + dof_index] = np.clip(
                trial[:, 7 + dof_index], lower, upper
            )
        final_bad_frames = [
            frame_id
            for frame_id in range(len(trial))
            if min_distance(trial[frame_id]) < 0.0
        ]
        if not final_bad_frames:
            break
        for frame_id in final_bad_frames:
            trial_qpos = trial[frame_id]
            initial = trial_qpos[7:] - deltas[frame_id]
            result = minimize(
                objective,
                initial + deltas[frame_id],
                args=(initial, trial_qpos[:3], trial_qpos[3:7]),
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": 35,
                    "ftol": 1e-9,
                    "maxls": 20,
                    "eps": 0.003,
                },
            )
            deltas[frame_id] = result.x - initial
            trial[frame_id, 7:] = np.clip(
                base_qpos[frame_id, 7:] + deltas[frame_id],
                np.asarray(bounds)[:, 0],
                np.asarray(bounds)[:, 1],
            )

    # 优化器和平滑后可能留下毫米级残余；用最小单关节修正清零。
    trial = base_qpos.copy()
    trial[:, 7:] += deltas
    for dof_index, (lower, upper) in enumerate(bounds):
        trial[:, 7 + dof_index] = np.clip(trial[:, 7 + dof_index], lower, upper)
    residual_bad_frames = [
        frame_id
        for frame_id in range(len(trial))
        if min_distance(trial[frame_id]) < 0.0
    ]
    repaired_frames = 0
    for frame_id in residual_bad_frames:
        candidates = []
        for dof_index in range(len(bounds)):
            for delta in (0.002, 0.005, 0.010, -0.002, -0.005, -0.010):
                candidate = trial[frame_id].copy()
                lower, upper = bounds[dof_index]
                candidate[7 + dof_index] = np.clip(
                    candidate[7 + dof_index] + delta, lower, upper
                )
                distance = min_distance(candidate)
                if distance >= 0.0:
                    candidates.append((abs(delta), -distance, dof_index, delta))
        if candidates:
            _, _, dof_index, delta = min(candidates)
            lower, upper = bounds[dof_index]
            deltas[frame_id, dof_index] = np.clip(
                deltas[frame_id, dof_index] + delta, lower, upper
            )
            trial[frame_id, 7 + dof_index] = np.clip(
                trial[frame_id, 7 + dof_index] + delta, lower, upper
            )
            repaired_frames += 1

    trial = base_qpos.copy()
    trial[:, 7:] += deltas
    for dof_index, (lower, upper) in enumerate(bounds):
        trial[:, 7 + dof_index] = np.clip(trial[:, 7 + dof_index], lower, upper)
    minimum = 1e9
    penetrations = {}
    penetrating_frames = 0
    for frame_id in range(len(trial)):
        minimum_frame, frame_penetrations = min_distance(
            trial[frame_id], return_penetrations=True
        )
        minimum = min(minimum, minimum_frame)
        if minimum_frame < 0.0:
            penetrating_frames += 1
        for key, distance in frame_penetrations.items():
            penetrations[key] = min(penetrations.get(key, 1e9), distance)
    print(
        f"  self-collision: iterations={resolved_iterations}, "
        f"min_distance={minimum:.5f}, penetrating_frames={penetrating_frames}, "
        f"residual_repaired_frames={repaired_frames}"
    )
    return trial, {
        "iterations": resolved_iterations,
        "min_distance": float(minimum),
        "penetrating_pair_count": len(penetrations),
        "residual_repaired_frames": repaired_frames,
    }


def retarget_file(
    csv_path,
    out_dir,
    target_xml,
    fps,
    csv_fps,
    method,
    hybrid_iterations,
    arm_scale,
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
    ground_mode,
    ground_clearance,
    foot_ground_offset,
    ground_percentile,
    max_ground_lowering,
    root_z_offset,
    ground_smooth_window,
    left_shoulder_roll_offset,
    right_shoulder_roll_offset,
    resolve_self_collision,
    max_joint_velocity,
):
    source_model = mj.MjModel.from_xml_path(str(ROOT / "tools" / "retargeting" / "gmr_lens110" / "assets" / "booster_k1" / "K1_serial.xml"))
    source_qpos = load_k1_csv_qpos(csv_path, source_model)
    target_model_path = pathlib.Path(target_xml)
    target_model = mj.MjModel.from_xml_path(str(target_model_path))

    if method == "hybrid":
        qpos_array = hybrid_map_k1_to_lens110(
            source_model,
            target_model,
            source_qpos,
            hybrid_iterations,
            arm_scale,
            stance_width_offset,
            hip_yaw_out_offset,
        )
    else:
        qpos_array = direct_map_k1_to_lens110(
            source_model,
            target_model,
            source_qpos,
            stance_width_offset,
            hip_yaw_out_offset,
        )

    dof_names = qpos_dof_names(target_model)
    apply_left_shoulder_roll_offset(
        target_model,
        qpos_array,
        left_shoulder_roll_offset,
    )
    apply_right_shoulder_roll_offset(
        target_model,
        qpos_array,
        right_shoulder_roll_offset,
    )
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
    collision_pairs = []
    collision_stats = None
    if resolve_self_collision:
        collision_pairs = build_nonadjacent_geom_pairs(target_model)
        qpos_array, collision_stats = resolve_self_collisions(
            target_model,
            qpos_array,
            collision_pairs,
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
    if root_xy_scale != 1.0:
        origin = qpos_array[:1, :2]
        qpos_array[:, :2] = origin + root_xy_scale * (qpos_array[:, :2] - origin)
    limit_joint_speeds(target_model, qpos_array, fps, max_joint_velocity)
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
    clip_lens110_joint_limits(qpos_array, target_model, dof_names)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{csv_path.stem}_lens110_{method}"
    pkl_path = out_dir / f"{stem}.pkl"
    txt_path = out_dir / f"{stem}.txt"
    motion_data = {
        "fps": float(fps),
        "root_pos": qpos_array[:, :3],
        "root_rot": qpos_array[:, 3:7][:, [1, 2, 3, 0]],
        "dof_pos": qpos_array[:, 7:],
        "dof_names": dof_names,
        "local_body_pos": None,
        "link_body_list": None,
        "source_file": str(csv_path),
        "source_robot": "booster_k1",
        "target_model_file": str(target_model_path),
        "retarget_method": method,
        "arm_scale": float(arm_scale),
        "left_shoulder_roll_offset": float(left_shoulder_roll_offset),
        "right_shoulder_roll_offset": float(right_shoulder_roll_offset),
        "self_collision_resolved": bool(resolve_self_collision),
        "max_joint_velocity": float(max_joint_velocity),
    }
    with pkl_path.open("wb") as file:
        pickle.dump(motion_data, file)
    convert_lens110_gmr_to_legged_lab(str(pkl_path), str(txt_path), fps=fps)

    dof_pos = reorder_dof(motion_data["dof_pos"], dof_names, TRAIN_JOINT_ORDER)
    npy_path = out_dir / f"{stem}.npy"
    np.save(npy_path, dof_pos.astype(np.float32))
    csv_root_pos, csv_root_rot, csv_dof_pos = resample_poses(
        motion_data["root_pos"],
        motion_data["root_rot"],
        dof_pos,
        fps,
        csv_fps,
    )
    csv_path_out = out_dir / f"{stem}_{int(round(csv_fps))}hz.csv"
    write_csv_28(csv_root_pos, csv_root_rot, csv_dof_pos, csv_path_out)
    report_joint_limits(csv_dof_pos, TRAIN_JOINT_ORDER, str(target_model_path))

    print(
        f"  root_z range: [{qpos_array[:, 2].min():.4f}, {qpos_array[:, 2].max():.4f}], "
        f"root_xy_scale={root_xy_scale:.3f}, arm_scale={arm_scale:.3f}"
    )
    if stance_stats is not None:
        print(
            "  stance: corrected_frames="
            f"{stance_stats['corrected_frames']}, "
            f"lat_sep before/after mean="
            f"{stance_stats['before_mean']:.3f}/{stance_stats['after_mean']:.3f}"
        )
    if ground_stats is not None:
        print(
            "  grounding: "
            f"mode={ground_mode}, foot_min_before={ground_stats['foot_min_before']:.3f}, "
            f"ground_offset={ground_stats['ground_offset']:.3f}"
        )
    return csv_path_out


def parse_args():
    parser = argparse.ArgumentParser(description="K1 CSV -> lens110 动作")
    parser.add_argument("--csv", required=True, type=pathlib.Path)
    parser.add_argument("--out_dir", required=True, type=pathlib.Path)
    parser.add_argument("--target_xml", type=pathlib.Path, default=DEFAULT_TARGET_XML)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--csv_fps", type=float, default=50.0)
    parser.add_argument("--method", choices=["direct", "hybrid"], default="hybrid")
    parser.add_argument("--hybrid_iterations", type=int, default=4)
    parser.add_argument("--arm_scale", type=float, default=0.72)
    parser.add_argument("--stance_width_offset", type=float, default=0.06)
    parser.add_argument("--hip_yaw_out_offset", type=float, default=0.0)
    parser.add_argument("--leg_motion_scale", type=float, default=0.5)
    parser.add_argument("--hip_yaw_scale", type=float, default=1.0)
    parser.add_argument("--ankle_roll_scale", type=float, default=0.0)
    parser.add_argument(
        "--resolve_self_collision", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--torso_yaw_scale", type=float, default=1.0)
    parser.add_argument("--lower_body_smooth_window", type=int, default=1)
    parser.add_argument("--min_stance_width", type=float, default=0.10)
    parser.add_argument("--stance_correction_gain", type=float, default=3.0)
    parser.add_argument("--stance_correction_iterations", type=int, default=3)
    parser.add_argument("--max_stance_correction_step", type=float, default=0.12)
    parser.add_argument("--root_xy_scale", type=float, default=0.72)
    parser.add_argument(
        "--ground_mode", choices=["constant", "per_frame", "smooth", "off"], default="per_frame"
    )
    parser.add_argument("--ground_clearance", type=float, default=0.02)
    parser.add_argument("--foot_ground_offset", type=float, default=0.0)
    parser.add_argument("--ground_percentile", type=float, default=5.0)
    parser.add_argument("--max_ground_lowering", type=float, default=0.06)
    parser.add_argument("--root_z_offset", type=float, default=0.0)
    parser.add_argument("--ground_smooth_window", type=int, default=5)
    parser.add_argument("--left_shoulder_roll_offset", type=float, default=0.06)
    parser.add_argument("--right_shoulder_roll_offset", type=float, default=-0.20)
    parser.add_argument("--max_joint_velocity", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    output = retarget_file(
        args.csv,
        args.out_dir,
        args.target_xml,
        args.fps,
        args.csv_fps,
        args.method,
        args.hybrid_iterations,
        args.arm_scale,
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
        args.ground_mode,
        args.ground_clearance,
        args.foot_ground_offset,
        args.ground_percentile,
        args.max_ground_lowering,
        args.root_z_offset,
        args.ground_smooth_window,
        args.left_shoulder_roll_offset,
        args.right_shoulder_roll_offset,
        args.resolve_self_collision,
        args.max_joint_velocity,
    )
    print(f"完成: {output}")


if __name__ == "__main__":
    main()
