"""播放 GMR lens110 pkl 动作 (MuJoCo 交互窗口, 带进度条和按键调速)。

pkl 里 root_rot 为 xyzw (GMR/batch_bvh 保存约定), 播放时转回 wxyz 给 MuJoCo。

用法:
    python play_lens110_pkl.py --pkl /tmp/lens110_full/pkl/tangbohushuoDJ.pkl
    python play_lens110_pkl.py --pkl xxx.pkl --speed 0.7   # 慢放
"""

import argparse
import os
import pickle

import numpy as np

from lens110_player import play_motion
from path_utils import lens110_mjcf

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MJCF = str(lens110_mjcf())


def main():
    parser = argparse.ArgumentParser(description="播放 lens110 GMR pkl 动作")
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--mjcf", default=DEFAULT_MJCF)
    parser.add_argument("--fps", type=float, default=None, help="播放帧率, 默认用 pkl 里的 fps")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度倍率, 0.5=半速")
    parser.add_argument("--loop", action="store_true", default=True)
    parser.add_argument("--fix_root", action="store_true", help="固定 root, 只看关节运动")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        motion = pickle.load(f)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    fps = args.fps or float(motion.get("fps", 30.0))
    n = len(root_pos)

    print(f"[pkl] frames={n} fps={fps:.1f} root_pos={root_pos.shape} "
          f"root_rot(xyzw)={root_rot_xyzw.shape} dof={dof_pos.shape}")
    print(f"      root z range: [{root_pos[:,2].min():.3f}, {root_pos[:,2].max():.3f}]")
    print(f"      dof range: [{dof_pos.min():.3f}, {dof_pos.max():.3f}]")

    play_motion(
        root_pos,
        root_rot_xyzw,
        dof_pos,
        fps,
        args.mjcf,
        speed=args.speed,
        loop=args.loop,
        fix_root=args.fix_root,
    )


if __name__ == "__main__":
    main()
