# Lens110 专用重定向脚本

## 入口脚本

| 脚本 | 用途 |
|---|---|
| `convert_bvh_to_lens110.py` | BVH -> Lens110 pkl/txt/npy/CSV 一键入口 |
| `batch_bvh_to_lens110.py` | 批量 BVH 重定向，默认从 `datasets/dance/raw_bvh/original_bvh` 读 |
| `batch_g1_csv_to_lens110.py` | G1 CSV -> Lens110 |
| `batch_k1_csv_to_lens110.py` | Booster K1 CSV -> Lens110 |
| `batch_smplx_to_lens110.py` | SMPL-X -> Lens110 |
| `lens110_pkl_to_training.py` | 已有 Lens110 pkl -> training npy/部署 CSV |
| `merge_upper_lower.py` | 上半身 pkl 与下半身 CSV 按真实时长合并 |
| `fix_hand_collision.py` | 用真实可见 elbow mesh 检查/修正手部穿模 |
| `fix_lens110_arm_swap.py` | 修正历史 CSV 的肩 roll/yaw 交换 |
| `play_lens110_pkl.py` / `play_lens110_csv.py` | MuJoCo 动作回放 |
| `verify_coord_transform.py` / `verify_lens110_retarget.py` | 坐标/重定向快速验证 |

## 数据位置

- 原始 BVH：`datasets/dance/raw_bvh/original_bvh/`；
- 全身处理动作：`projects/01_dance_whole_body/data/processed/retargeted_actions/`；
- 半身合并/upper-lower 相关动作：`projects/02_dance_half_body/data/processed/retargeted_actions/`；
- 临时转换产物：推荐使用 `/tmp/lens110_*`，验证后再归档；
- 默认 Lens110 MJCF：由 `path_utils.lens110_mjcf()` 定位，不再依赖固定的 `/home/lbot` 或旧 `lens110RL`。

## 关键契约

```text
GMR pkl/CSV root quaternion: xyzw
MuJoCo qpos free-joint quaternion: wxyz
deployment CSV: 3 root position + 4 xyzw quaternion + 21 dof
training full-frame: 120 Hz
deployment CSV: normally 50 Hz
```

训练全帧不要为了匹配 CSV 行数而重采样到 50 Hz；这会改变动作物理时长。每次后处理都应生成一份 quality
report，并把来源、fps、joint order、地面模式和修复参数记录在项目数据目录。
