"""Drive the simulated Zang with zang_ik: squat, weight shift, single-leg lift, arm reach, waist turn.

    ../.venv/bin/python demo_mujoco.py            # headless run, prints stability report
    ../.venv/bin/python demo_mujoco.py --view     # watch it in the MuJoCo viewer
    ../.venv/bin/python demo_mujoco.py --render frames.png   # contact sheet of the motion
"""
import argparse
import math
import os
import time

import mujoco
import numpy as np

from zang_ik import SIDES, Zang

MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zang_description", "mjcf", "scene.xml")
RATE = 50  # Hz, IK / setpoint rate


def keyframes(z, feet, hands0, pelvis0, com0):
    """(seconds, CoM xy, pelvis z, {side: lift}, {side: hand offset}, waist yaw) waypoints."""
    left_y, right_y = feet["left"][0][1], feet["right"][0][1]
    c = com0[:2]
    zc, zs = pelvis0[2], pelvis0[2] - 0.02
    none, reach = {}, {"left": np.array([0.05, 0.0, 0.05]), "right": np.array([0.05, 0.0, 0.05])}
    return [
        (1.0, c, zc, {}, none, 0.0),                                  # hold home
        (1.5, c, zs, {}, none, 0.0),                                  # squat 20 mm
        (1.5, c, zc, {}, none, 0.0),                                  # stand back to home height
        (2.0, [c[0], left_y * 0.85], zc, {}, none, 0.0),              # shift CoM over the left foot
        (1.5, [c[0], left_y * 0.85], zc, {"right": 0.015}, none, 0.0),  # lift right foot 15 mm
        (1.0, [c[0], left_y * 0.85], zc, {"right": 0.015}, none, 0.0),  # balance on one leg
        (1.5, [c[0], left_y * 0.85], zc, {}, none, 0.0),              # put it down
        (2.0, c, zc, {}, none, 0.0),                                  # back to the middle
        (1.5, c, zc, {}, reach, 0.0),                                 # reach forward/up with both hands
        (1.5, c, zc, {}, reach, 0.6),                                 # turn the waist
        (2.0, c, zc, {}, none, 0.0),                                  # home
    ]


def plan(z, m):
    key = m.key("home").id
    pelvis0 = m.key_qpos[key][:3].copy()
    F = z.fk(z.home(), pelvis0)
    feet = {s: (z.sole_pos(F, s), 0.0) for s in SIDES}
    hands0 = {s: F["chest"][:3, :3].T @ (z.hand_pos(F, s) - F["chest"][:3, 3]) for s in SIDES}  # chest frame
    com0 = z.com(F)
    wps = keyframes(z, feet, hands0, pelvis0, com0)
    traj, prev = [], (com0[:2], pelvis0[2], {}, {}, 0.0)
    for dur, com_xy, pz, lift, hand_off, yaw in wps:
        for k in range(int(dur * RATE)):
            a = 0.5 - 0.5 * math.cos(math.pi * (k + 1) / (dur * RATE))  # smooth step
            mix = lambda x0, x1: (1 - a) * np.asarray(x0, float) + a * np.asarray(x1, float)
            f = {s: (feet[s][0] + [0, 0, mix(prev[2].get(s, 0), lift.get(s, 0))], 0.0) for s in SIDES}
            off = {s: mix(prev[3].get(s, np.zeros(3)), hand_off.get(s, np.zeros(3))) for s in SIDES}
            w = float(mix(prev[4], yaw))
            q, pp = z.balance_ik(mix(prev[0], com_xy), float(mix(prev[1], pz)), f)
            if any(np.any(o) for o in off.values()):  # arms: chest-frame targets relative to the home hand position
                for s in SIDES:
                    q.update(z.arm_ik(s, hands0[s] + off[s]))
            q["waist_yaw"] = w
            traj.append(z.full_q(q))
        prev = (com_xy, pz, lift, hand_off, yaw)
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--render", metavar="PNG")
    args = ap.parse_args()

    z = Zang()
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    traj = plan(z, m)
    act = [m.actuator(i).name for i in range(m.nu)]
    pelvis = m.body("pelvis").id
    floor = m.geom("floor").id
    steps_per = int(round(1 / (RATE * m.opt.timestep)))
    viewer = mujoco.viewer.launch_passive(m, d) if args.view else None
    renderer = mujoco.Renderer(m, 360, 320) if args.render else None
    frames, max_tilt, min_feet_contacts = [], 0.0, {}
    for i, q in enumerate(traj):
        d.ctrl[:] = [q[n] for n in act]
        for _ in range(steps_per):
            mujoco.mj_step(m, d)
        up = d.xmat[pelvis].reshape(3, 3)[:, 2]
        max_tilt = max(max_tilt, math.degrees(math.acos(min(1.0, up[2]))))
        touching = {m.body(m.geom_bodyid[g]).name for c in d.contact[:d.ncon] for g in (c.geom1, c.geom2) if c.geom1 == floor or c.geom2 == floor} - {"world"}
        min_feet_contacts[i] = touching
        if viewer:
            viewer.sync()
            time.sleep(1 / RATE)
        if renderer and i % (RATE * 2) == RATE:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [d.xpos[pelvis][0], d.xpos[pelvis][1], 0.15]
            cam.distance, cam.azimuth, cam.elevation = 0.75, 150, -15
            renderer.update_scene(d, cam)
            frames.append(renderer.render())
    lifted = [i for i, t in min_feet_contacts.items() if "right_foot" not in t]
    print(f"{len(traj)/RATE:.1f} s of motion, max pelvis tilt {max_tilt:.1f} deg, "
          f"right foot off the ground for {len(lifted)/RATE:.2f} s, final pelvis z {d.xpos[pelvis][2]*1000:.1f} mm")
    print("FELL" if max_tilt > 30 else "stayed upright")
    if renderer:
        from PIL import Image
        Image.fromarray(np.hstack(frames)).save(args.render)
        print("wrote", args.render)
    if viewer:
        viewer.close()


if __name__ == "__main__":
    import mujoco.viewer  # noqa: F401  (only needed for --view, but harmless)
    main()
