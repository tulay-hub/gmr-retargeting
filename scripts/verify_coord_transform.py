"""验证坐标系转换是否正确"""
import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from general_motion_retargeting.utils.lafan1 import load_bvh_file
from path_utils import repository_root

BVH_FILE = str(repository_root() / "datasets" / "dance" / "raw_bvh" / "original_bvh" / "tangbohushuoDJ.bvh")

def main():
    # 加载BVH数据
    frames, human_height = load_bvh_file(BVH_FILE, format="nokov")
    frame0 = frames[0]
    
    print("=" * 60)
    print("1. 原始BVH数据（lafan1.py处理后）")
    print("=" * 60)
    
    key_bodies = ["Hips", "LeftUpLeg", "RightUpLeg", "LeftArm", "RightArm"]
    for body in key_bodies:
        pos = frame0[body][0]
        print(f"{body:<15} pos: [{pos[0]:+8.3f} {pos[1]:+8.3f} {pos[2]:+8.3f}]")
    
    print("\n推断坐标系：")
    print(f"  Hips高度: {frame0['Hips'][0][2]:.3f}m (应该是~0.95m)")
    print(f"  LeftUpLeg X: {frame0['LeftUpLeg'][0][0]:+.3f} (正=左?)")
    print(f"  LeftUpLeg Y: {frame0['LeftUpLeg'][0][1]:+.3f} (负=前?)")
    
    print("\n" + "=" * 60)
    print("2. 坐标系转换测试")
    print("=" * 60)
    
    # 测试绕Z轴+90度
    rot_z90 = R.from_euler("z", 90, degrees=True)
    
    print("\n绕Z轴+90度变换：")
    print("  (1,0,0) →", rot_z90.apply([1,0,0]))
    print("  (0,1,0) →", rot_z90.apply([0,1,0]))
    print("  (0,0,1) →", rot_z90.apply([0,0,1]))
    
    # 应用变换
    print("\n变换后的BVH数据：")
    for body in key_bodies:
        pos = frame0[body][0]
        new_pos = rot_z90.apply(pos)
        print(f"{body:<15} [{pos[0]:+8.3f} {pos[1]:+8.3f} {pos[2]:+8.3f}] → [{new_pos[0]:+8.3f} {new_pos[1]:+8.3f} {new_pos[2]:+8.3f}]")
    
    print("\n期望的机器人坐标系：")
    print("  X前, Y左, Z上")
    print("  Hips应该在(0, 0, ~0.95)")
    print("  LeftUpLeg应该在(0, +, ~0.9) - Y正方向（左）")
    
    print("\n" + "=" * 60)
    print("3. 验证")
    print("=" * 60)
    
    hips_new = rot_z90.apply(frame0["Hips"][0])
    left_leg_new = rot_z90.apply(frame0["LeftUpLeg"][0])
    
    print(f"Hips变换后: [{hips_new[0]:+.3f}, {hips_new[1]:+.3f}, {hips_new[2]:+.3f}]")
    print(f"  期望: [0, 0, ~0.95]")
    print(f"  正确? {abs(hips_new[0]) < 0.01 and abs(hips_new[1]) < 0.01 and abs(hips_new[2] - 0.945) < 0.01}")
    
    print(f"\nLeftUpLeg变换后: [{left_leg_new[0]:+.3f}, {left_leg_new[1]:+.3f}, {left_leg_new[2]:+.3f}]")
    print(f"  期望: [0, +, ~0.9] - Y正方向（左）")
    print(f"  Y>0? {left_leg_new[1] > 0}")

if __name__ == "__main__":
    main()
