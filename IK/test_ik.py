"""Checks for zang_ik: FK against MuJoCo, IK round trips, whole-body, balance and numeric IK.

Run: ../.venv/bin/python test_ik.py   (or pytest)
"""
import math
import os

import numpy as np

from zang_ik import HOME, SIDES, Zang, rpy, rot, EX

z = Zang()
rng = np.random.default_rng(0)
MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zang_description", "mjcf", "scene.xml")


def random_leg(side):
    return {f"{side}_hip_roll": rng.uniform(-0.4, 0.4), f"{side}_hip_pitch": rng.uniform(-1.2, 0.5),
            f"{side}_knee_pitch": rng.uniform(-1.2, 0.5), f"{side}_ankle_roll": rng.uniform(-0.4, 0.4)}


def test_fk_matches_mujoco():
    import mujoco
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    worst = 0.0
    for _ in range(20):
        q = z.full_q({n: rng.uniform(-1.0, 1.0) for n in z.actuated})
        d.qpos[:] = 0
        d.qpos[3] = 1
        for n, v in q.items():
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
        mujoco.mj_kinematics(m, d)
        F = z.fk(q, d.xpos[m.body(z.root).id], d.xmat[m.body(z.root).id].reshape(3, 3))
        for link, T in F.items():
            b = m.body(link).id
            worst = max(worst, np.abs(T[:3, 3] - d.xpos[b]).max(), np.abs(T[:3, :3] - d.xmat[b].reshape(3, 3)).max())
    assert worst < 1e-9, worst
    return f"FK vs MuJoCo max error {worst:.1e}"


def test_leg_ik_round_trip():
    worst = 0.0
    for _ in range(500):
        side = SIDES[rng.integers(2)]
        q = random_leg(side)
        F = z.fk(q)
        target, roll = z.sole_pos(F, side), q[f"{side}_hip_roll"] + q[f"{side}_ankle_roll"]
        sol = z.leg_ik(side, target, roll)
        F2 = z.fk(sol)
        worst = max(worst, np.linalg.norm(z.sole_pos(F2, side) - target),
                    np.abs(F2[f"{side}_foot"][:3, :3] - F[f"{side}_foot"][:3, :3]).max())
    assert worst < 1e-9, worst
    return f"leg IK round trip (500 poses) max error {worst:.1e}"


def test_leg_ik_prefers_knee_forward():
    q = z.leg_ik("left", z.sole_pos(z.fk(HOME), "left"))
    assert abs(q["left_hip_pitch"] - HOME["left_hip_pitch"]) < 1e-9 and abs(q["left_knee_pitch"] - HOME["left_knee_pitch"]) < 1e-9
    F = z.fk(q)
    knee, hip, ankle = (F[f"left_{n}"][:3, 3] for n in ("knee", "thigh", "ankle"))
    assert knee[0] > (hip[0] + ankle[0]) / 2, "knee should point forward"
    return "leg IK recovers HOME with the knee forward"


def test_arm_ik_round_trip():
    worst = 0.0
    for _ in range(500):
        side = SIDES[rng.integers(2)]
        q = {f"{side}_{n}": rng.uniform(-1.4, 1.4) for n in ("shoulder_pitch", "shoulder_roll", "elbow")}
        F = z.fk(q)
        target = F["chest"][:3, :3].T @ (z.hand_pos(F, side) - F["chest"][:3, 3])
        sol = z.arm_ik(side, target, hint=q)
        F2 = z.fk(sol)
        worst = max(worst, np.linalg.norm(z.hand_pos(F2, side) - z.hand_pos(F, side)))
    assert worst < 1e-9, worst
    return f"arm IK round trip (500 poses) max error {worst:.1e}"


def test_whole_body_ik():
    feet = {"left": (np.array([0.01, 0.065, 0.0]), 0.0), "right": (np.array([-0.01, -0.06, 0.0]), 0.0)}
    pp, prpy = np.array([0.0, 0.01, 0.225]), (0.05, 0.0, 0.3)
    arms = {"waist_yaw": 0.2, "left_shoulder_pitch": -0.8, "left_shoulder_roll": 0.3, "left_elbow": 0.4,
            "right_shoulder_pitch": 0.5, "right_shoulder_roll": -0.2, "right_elbow": -0.6}
    F0 = z.fk(arms, pp, rpy(*prpy))
    hands = {s: z.hand_pos(F0, s) for s in SIDES}
    feet_w = {s: (rot([0, 0, 1], 0.3) @ p, r) for s, (p, r) in feet.items()}
    q = z.whole_body_ik(pp, prpy, feet_w, hands, waist_yaw=0.2)
    F = z.fk(q, pp, rpy(*prpy))
    err = max(max(np.linalg.norm(z.sole_pos(F, s) - feet_w[s][0]) for s in SIDES),
              max(np.linalg.norm(z.hand_pos(F, s) - hands[s]) for s in SIDES))
    up = max(abs(F[f"{s}_foot"][2, 2] - 1) for s in SIDES)
    assert err < 1e-9 and up < 1e-9, (err, up)
    return f"whole-body IK (pelvis rolled+yawed, waist turned) max error {err:.1e}, feet flat"


def test_balance_ik():
    feet = {"left": ([0, 0.058, 0], 0.0), "right": ([0, -0.058, 0], 0.0)}
    q, pp = z.balance_ik([0.0, 0.045], 0.22, feet)
    c = z.com(z.fk(q, pp))
    assert np.linalg.norm(c[:2] - [0.0, 0.045]) < 1e-5
    return f"balance IK puts CoM over left foot (pelvis shifted to y={pp[1]*1000:.1f} mm)"


def test_numeric_ik():
    target = z.hand_pos(z.fk({"left_shoulder_pitch": -0.6, "left_elbow": 0.5}), "left")
    q = z.numeric_ik({"left_hand": (z.hand_tip["left"], target)},
                     joints=["left_shoulder_pitch", "left_shoulder_roll", "left_elbow"])
    err = np.linalg.norm(z.hand_pos(z.fk(q), "left") - target)
    assert err < 1e-6, err
    return f"numeric IK converges ({err:.1e} m)"


def test_home_matches_mjcf_keyframe():
    import mujoco
    m = mujoco.MjModel.from_xml_path(MJCF)
    k = m.key("home").id
    for i in range(m.nu):
        assert math.isclose(m.key_ctrl[k][i], HOME.get(m.actuator(i).name, 0.0), abs_tol=1e-6), m.actuator(i).name
    return "HOME equals MJCF keyframe 'home'"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(f"ok  {fn()}")
