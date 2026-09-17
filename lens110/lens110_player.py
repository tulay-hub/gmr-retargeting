"""lens110 动作播放公共核心: 进度条 + 按键控制 + 视角控制。

按键 (MuJoCo 窗口获得焦点时):
    ,  减速 (x0.8)          .  加速 (x1.25)      1..9  直接跳到 x0.1 .. x1.0
    P                      暂停 / 继续
    R                      从头重播
    N  前进 5s             B  后退 5s
    U  上一帧              O  下一帧
    F  切换 跟随视角 / 自由视角
    V  重置视角 (对准当前机器人, 默认方位角/俯仰角)
    C  终端打印当前帧/速度/视角状态
    Esc                    退出

为什么要换键位: MuJoCo 3.x 自带的 simulate 窗口把
    Space = Play/Pause,  [ ] = Cycle cameras,  + - = Speed Up/Down,
    左右方向键 = Step Back/Forward
占为已用。旧版用 [ ] 调速时其实同时把相机切成了 Tracking(看起来像"锁定视角"),
用方向键还会让 viewer 自己 step 物理, 污染 kinematic 播放。
本播放器每帧把相机类型钉回自己记录的模式, 所以不会再被误锁。

播放节奏改用墙上时钟 (wall clock) 计算帧号, 倍速就是真倍速;
旧版每帧 sleep(1/fps) 会忽略渲染耗时, 实际速度比标称慢。

窗口四角显示: 帧号/进度条/时间、倍速与视角模式、当前关节速度、root 高度。
"""

import time

import mujoco
import mujoco.viewer
import numpy as np
from tqdm import tqdm


# GLFW keycode (字母以大写 ASCII 传入)
K_SPACE = 32
K_COMMA = 44
K_PERIOD = 46
K_0 = 48  # '0'..'9' -> 48..57
K_B = 66
K_C = 67
K_F = 70
K_N = 78
K_O = 79
K_P = 80
K_R = 82
K_U = 85
K_V = 86
K_ESCAPE = 256

DEFAULT_DISTANCE = 2.4
DEFAULT_ELEVATION = -14.0
DEFAULT_AZIMUTH = 135.0


def _body_id(model, *cands):
    for name in cands:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return bid
    return 1


def play_motion(
    root_pos,
    root_rot_xyzw,
    dof_pos,
    fps,
    mjcf,
    speed=1.0,
    loop=True,
    fix_root=False,
    follow=True,
):
    """播放一段 lens110 动作 (root_pos + root_rot xyzw + dof_pos)。

    root_rot 按 pkl/CSV 约定为 xyzw, 内部转 wxyz 给 MuJoCo。
    follow=True 时相机默认跟随 pelvis (动作有位移时不会跟丢)。
    """
    root_pos = np.asarray(root_pos, dtype=np.float64)
    root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float64)
    dof_pos = np.asarray(dof_pos, dtype=np.float64)
    n = len(root_pos)

    if fix_root:
        root_pos = np.zeros_like(root_pos)
        root_pos[:, 2] = 0.95
        root_rot_xyzw[:] = [0.0, 0.0, 0.0, 1.0]
        print("[fix_root] 已固定 root")

    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)

    # 关节速度 (rad/s), 用于 HUD 与超限提示
    vel = np.zeros_like(dof_pos)
    vel[1:] = np.abs(np.diff(dof_pos, axis=0)) * float(fps)
    if n > 1:
        vel[0] = vel[1]

    track_body = _body_id(model, "pelvis", "base_link", "torso_yaw_link")
    st = {
        "speed": float(speed),
        "paused": False,
        "frame": 0.0,
        "follow": bool(follow),
        "t_anchor": time.time(),
        "f_anchor": 0.0,
    }

    def reanchor(new_frame=None):
        st["t_anchor"] = time.time()
        st["f_anchor"] = st["frame"] if new_frame is None else float(new_frame)

    def on_key(keycode):
        if keycode == K_COMMA:
            st["speed"] = max(0.05, st["speed"] * 0.8)
            reanchor()
        elif keycode == K_PERIOD:
            st["speed"] = min(10.0, st["speed"] * 1.25)
            reanchor()
        elif K_0 + 1 <= keycode <= K_0 + 9:
            st["speed"] = (keycode - K_0) * 0.1
            reanchor()
        elif keycode == K_P or keycode == K_SPACE:
            st["paused"] = not st["paused"]
            if not st["paused"]:
                reanchor()
        elif keycode == K_R:
            reanchor(0)
        elif keycode == K_N:
            reanchor(min(n - 1, st["frame"] + 5.0 * fps))
        elif keycode == K_B:
            reanchor(max(0.0, st["frame"] - 5.0 * fps))
        elif keycode == K_U:
            reanchor(max(0.0, st["frame"] - 1))
        elif keycode == K_O:
            reanchor(min(n - 1, st["frame"] + 1))
        elif keycode == K_F:
            st["follow"] = not st["follow"]
        elif keycode == K_V:
            pass  # 主循环里检测到 flag 后处理, 这里直接置位
            st["reset_view"] = True
        elif keycode == K_C:
            i = int(st["frame"])
            print(f"[player] frame {i}/{n}  t {i / fps:.1f}s / {n / fps:.1f}s  "
                  f"speed {st['speed']:.2f}x  cam={'跟随' if st['follow'] else '自由'}  "
                  f"{'暂停' if st['paused'] else '播放'}")
        else:
            return
        if not st["paused"] and keycode not in (K_P, K_SPACE):
            pass

    print("[player] 按键: , . 调速 | 1-9 定倍速 | P 暂停 | R 重播 | B/N +-5s | U/O 单帧 | F 视角 | V 重置视角 | C 状态 | Esc 退出")
    print(f"[player] 当前速度 {st['speed']:.2f}x, 共 {n} 帧 / {n / fps:.1f}s, "
          f"视角: {'跟随' if st['follow'] else '自由'}")

    def apply_cam(viewer):
        cam = viewer.cam
        want = mujoco.mjtCamera.mjCAMERA_TRACKING if st["follow"] else mujoco.mjtCamera.mjCAMERA_FREE
        if cam.type != want:
            cam.type = want
        if st["follow"]:
            cam.trackbodyid = track_body
            if cam.distance <= 0.1:
                cam.distance = DEFAULT_DISTANCE
        if st.pop("reset_view", False):
            if st["follow"]:
                cam.azimuth = DEFAULT_AZIMUTH
                cam.elevation = DEFAULT_ELEVATION
                cam.distance = DEFAULT_DISTANCE
            else:
                cam.lookat = data.xpos[track_body].copy()
                cam.azimuth = DEFAULT_AZIMUTH
                cam.elevation = DEFAULT_ELEVATION
                cam.distance = DEFAULT_DISTANCE

    last_hud = 0.0
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        pbar = tqdm(total=n, desc="playing", unit="frame", dynamic_ncols=True, leave=False)
        apply_cam(viewer)
        while viewer.is_running():
            if not st["paused"]:
                st["frame"] = st["f_anchor"] + (time.time() - st["t_anchor"]) * fps * st["speed"]
                if st["frame"] >= n:
                    if loop:
                        reanchor(0)
                        pbar.reset()
                    else:
                        break
            i = int(min(max(st["frame"], 0), n - 1))

            data.qpos[:3] = root_pos[i]
            data.qpos[3:7] = root_rot_xyzw[i][[3, 0, 1, 2]]  # xyzw -> wxyz
            data.qpos[7:7 + dof_pos.shape[1]] = dof_pos[i]
            mujoco.mj_forward(model, data)

            pbar.update(0)
            pbar.n = i
            pbar.refresh()

            now = time.time()
            if now - last_hud > 0.08:
                last_hud = now
                vmax = float(vel[i].max())
                jmax = int(np.argmax(vel[i]))
                mj = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jmax + 1) or f"joint{jmax}"
                over = "  [超速]" if vmax > 10.4 else ""
                hud = [
                    (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     f"Frame {i}/{n}", f"{i / fps:.1f}s / {n / fps:.1f}s"),
                    (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                     f"speed {st['speed']:.2f}x",
                     f"cam {'FOLLOW' if st['follow'] else 'FREE'}" + ("  PAUSED" if st["paused"] else "")),
                    (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                     f"max joint vel {vmax:.1f} rad/s{over}", f"{mj}"),
                    (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                     f"root z {data.qpos[2]:.3f} m", f"fps {fps:.0f}"),
                ]
                # set_texts 每次调用会整体替换, 所以四角文本要一次传整个列表
                try:
                    viewer.set_texts(hud)
                except Exception:
                    pass

            apply_cam(viewer)
            viewer.sync()
            # 让出 GIL, 避免忙等; 真正的节奏由上面的墙上时钟决定
            time.sleep(0.001)

        pbar.close()
    print("[done] 播放结束")
