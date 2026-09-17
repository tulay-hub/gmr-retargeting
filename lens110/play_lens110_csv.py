"""播放 lens110 28 列部署 CSV 动作 (MuJoCo 交互窗口, 带进度条和按键调速)。

CSV 格式与旧版一致: [root_pos(3) | root_rot xyzw(4) | 21 关节], 行末带逗号。

用法:
    python tools/retargeting/gmr_lens110/lens110/play_lens110_csv.py \
        --csv projects/01_dance_whole_body/data/processed/retargeted_actions/tangbohushuo_dance/training/tangbohushuoDJ_v2_100hz.csv
    python play_lens110_csv.py --csv xxx.csv --speed 0.7   # 慢放
"""

import argparse
import os

import numpy as np

from lens110_player import play_motion
from path_utils import lens110_mjcf

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MJCF = str(lens110_mjcf())


def load_csv(path):
    with open(path) as f:
        rows = [[float(x) for x in line.rstrip().rstrip(",").split(",")] for line in f if line.strip()]
    arr = np.array(rows)
    if arr.shape[1] != 28:
        raise RuntimeError(f"期望 28 列, 实际 {arr.shape[1]}")
    return arr


def main():
    parser = argparse.ArgumentParser(description="播放 lens110 28 列部署 CSV")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--mjcf", default=DEFAULT_MJCF)
    parser.add_argument("--fps", type=float, default=50.0, help="播放帧率, 默认 50 (部署 CSV 帧率)")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率, 0.5=半速")
    parser.add_argument("--loop", action="store_true", default=True)
    parser.add_argument("--fix_root", action="store_true", help="固定 root, 只看关节运动")
    args = parser.parse_args()

    csv = load_csv(args.csv)
    n = len(csv)
    print(f"[csv] frames={n} fps={args.fps} shape={csv.shape}")
    print(f"      root z range: [{csv[:,2].min():.3f}, {csv[:,2].max():.3f}]")
    print(f"      dof range: [{csv[:,7:].min():.3f}, {csv[:,7:].max():.3f}]")

    play_motion(
        csv[:, :3],
        csv[:, 3:7],
        csv[:, 7:28],
        args.fps,
        args.mjcf,
        speed=args.speed,
        loop=args.loop,
        fix_root=args.fix_root,
    )


if __name__ == "__main__":
    main()
