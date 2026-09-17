import argparse
import pickle

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_GMR_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

LEGGED_LAB_MOTION_JOINT_ORDER = [
    "torso_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]


def _quat_conjugate(quat_wxyz):
    result = quat_wxyz.copy()
    result[..., 1:] *= -1.0
    return result


def _quat_mul(quat_a, quat_b):
    aw, ax, ay, az = np.moveaxis(quat_a, -1, 0)
    bw, bx, by, bz = np.moveaxis(quat_b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _axis_angle_from_quat(quat_wxyz):
    quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    xyz = quat_wxyz[..., 1:]
    xyz_norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(xyz_norm, quat_wxyz[..., :1])
    axis = np.divide(xyz, xyz_norm, out=np.zeros_like(xyz), where=xyz_norm > 1e-8)
    return axis * angle


def _reorder_dofs(dof_pos, source_joint_order):
    if len(source_joint_order) != dof_pos.shape[1] and "root" in source_joint_order:
        source_joint_order = [name for name in source_joint_order if name != "root"]
    if len(source_joint_order) != dof_pos.shape[1]:
        raise ValueError(
            f"Expected {dof_pos.shape[1]} joint names, got {len(source_joint_order)}"
        )
    source_index = {name: index for index, name in enumerate(source_joint_order)}
    missing = [name for name in LEGGED_LAB_MOTION_JOINT_ORDER if name not in source_index]
    if missing:
        raise ValueError(f"Missing joints in GMR motion: {missing}")
    indices = [source_index[name] for name in LEGGED_LAB_MOTION_JOINT_ORDER]
    return dof_pos[:, indices]


def convert_lens110_gmr_to_legged_lab(input_pkl, output_txt, fps=None):
    with open(input_pkl, "rb") as file:
        motion_data = pickle.load(file)

    motion_fps = float(fps if fps is not None else motion_data.get("fps", 30.0))
    dt = 1.0 / motion_fps

    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]
    source_joint_order = motion_data.get("dof_names", DEFAULT_GMR_JOINT_ORDER)
    dof_pos = _reorder_dofs(motion_data["dof_pos"], source_joint_order)

    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt
    delta_quat = _quat_mul(_quat_conjugate(root_rot[:-1]), root_rot[1:])
    root_ang_vel = _axis_angle_from_quat(delta_quat) / dt
    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt
    euler_angles = Rotation.from_quat(root_rot[:-1, [1, 2, 3, 0]]).as_euler("XYZ", degrees=False)
    euler_angles = np.unwrap(euler_angles, axis=0)

    frames = np.concatenate(
        (root_pos[:-1], euler_angles, dof_pos[:-1], root_lin_vel, root_ang_vel, dof_vel),
        axis=1,
    )

    with open(output_txt, "w") as file:
        file.write("{\n")
        file.write('"LoopMode": "Wrap",\n')
        file.write(f'"FrameDuration": {dt:.3f},\n')
        file.write('"EnableCycleOffsetPosition": true,\n')
        file.write('"EnableCycleOffsetRotation": true,\n')
        file.write('"MotionWeight": 0.5,\n\n')
        file.write('"Frames":\n[\n')
        for index, frame in enumerate(frames):
            values = ", ".join(f"{value:f}" for value in frame)
            suffix = "\n" if index == len(frames) - 1 else ",\n"
            file.write(f"  [{values}]{suffix}")
        file.write("]\n}")

    print(f"Successfully converted {input_pkl} to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_txt", type=str, required=True)
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

    convert_lens110_gmr_to_legged_lab(args.input_pkl, args.output_txt, args.fps)
