<p align="center"><a href="#zh">🇨🇳 中文</a> &nbsp;|&nbsp; <a href="#en">🇬🇧 English</a></p>
<a id="zh"></a>

# GMR Lens110 Retargeting

这是仓库中第一套重定向工具，实体目录为 `tools/retargeting/gmr_lens110`。它在 GMR 通用 IK 上增加了
Lens110 的二阶段适配、贴地、站距、限位和训练数据导出。

## 推荐流程

```text
BVH(nokov, 120 Hz)
  -> unitree_g1（GMR IK）
  -> lens110_21dof（direct/hybrid 映射）
  -> root/脚底/站距/关节限位/手臂平滑后处理
  -> pkl + txt
  -> 120 Hz training npy + 50 Hz deployment CSV
```

推荐 `hybrid`：腿部用同名关节，手臂使用上肢 IK 追踪。直接 BVH->Lens110 的旧配置只作为历史对照，原因是
坐标手性和肩部 roll 方向不能靠一个四元数偏移解决。

## 快速开始

在仓库根目录、已激活 `gmr` 环境中：

```bash
conda activate gmr
python tools/retargeting/gmr_lens110/lens110/convert_bvh_to_lens110.py \
  --bvh datasets/dance/raw_bvh/original_bvh/tangbohushuoDJ.bvh \
  --out_dir /tmp/lens110_tangbohushuoDJ \
  --method hybrid \
  --retarget_fps 120 \
  --csv_fps 50
```

默认 MJCF 由 `lens110/path_utils.py` 自动定位到共享训练资产；也可以显式传 `--target_xml`。

## 输出验收

- pkl 的 root quaternion 保持 GMR `xyzw`；
- training npy 为 `(N, 21)`，保持 120 Hz；
- deployment CSV 为 `root_pos(3) + root_rot_xyzw(4) + 21 dof`，每行尾逗号；
- 检查 21 个关节限位、root 姿态、脚底贴地、最小站距、速度和可见手部 mesh；
- 通过回放后，才把产物移动/登记到对应项目 `projects/*/data/processed` 或 `exports`。

Lens110 专用脚本列表和路径约定见 [`lens110/README.md`](lens110/README.md)。

<a id="en"></a>

## English

This repository contains the GMR motion-retargeting workflow. Its primary path is human-motion input such as BVH, retargeted through a supported robot model, and exported as local Lens110 21-DoF motion data. It also contains IK profiles, robot assets, MuJoCo inspection tools, and conversion utilities.

Use the local `gmr` environment and pass explicit input, target XML, and output paths. Before a clip becomes training data, verify coordinate frames, quaternion order, joint order, FPS, ground contacts, limits, finite values, and replay behavior. A robot-model replay is simulation evidence only.
