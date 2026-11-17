"""Verify the serial-parallel hybrid leg model against the true closed loops.

1. Kinematic: for each servo setting, solve the real four-bar loops from the CAD pin positions
   (MJCF loop sites) and compare with the ideal-parallelogram mimic relations used by the model.
2. Dynamic: simulate the true closed chain in MuJoCo (pins free, loops closed by `connect`
   constraints instead of joint couplings) and compare the settled pin angles with the mimic values.

    ../.venv/bin/python verify_linkage.py
"""
import os
import re

import mujoco
import numpy as np
from scipy.optimize import least_squares

MJCF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zang_description", "mjcf")
MIMIC = {"knee_upper": ("hip_pitch", -1), "thigh_shell": ("hip_pitch", 1),
         "knee_lower": ("knee_pitch", -1), "shin_shell": ("knee_pitch", -1)}


def set_mimic(m, d):
    for e in range(m.neq):
        if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
            d.qpos[m.jnt_qposadr[m.eq_obj1id[e]]] = m.eq_data[e][1] * d.qpos[m.jnt_qposadr[m.eq_obj2id[e]]]


def kinematic_check(m, side, qh, qk):
    """Exact loop closure for one leg; returns (pin angle errors vs mimic [rad], sole error [m], foot pitch [rad])."""
    d = mujoco.MjData(m)
    adr = lambda j: m.jnt_qposadr[m.joint(f"{side}_{j}").id]
    d.qpos[3] = 1
    d.qpos[adr("hip_pitch")], d.qpos[adr("knee_pitch")] = qh, qk
    set_mimic(m, d)
    mujoco.mj_kinematics(m, d)
    ideal = {j: d.qpos[adr(j)] for j in MIMIC}
    foot = m.body(f"{side}_foot").id
    sole_ideal, R_ideal = d.xpos[foot].copy(), d.xmat[foot].reshape(3, 3).copy()

    def closure(x, names, loop):
        for n, v in zip(names, x):
            d.qpos[adr(n)] = v
        mujoco.mj_kinematics(m, d)
        gap = d.site(f"{side}_{loop}_loop_a").xpos - d.site(f"{side}_{loop}_loop_b").xpos
        return gap[[0, 2]]  # planar loop: x/z

    up = least_squares(closure, [ideal["knee_upper"], ideal["thigh_shell"]], args=(("knee_upper", "thigh_shell"), "upper"), xtol=1e-14)
    lo = least_squares(closure, [ideal["knee_lower"], ideal["shin_shell"]], args=(("knee_lower", "shin_shell"), "lower"), xtol=1e-14)
    closure(up.x, ("knee_upper", "thigh_shell"), "upper")
    closure(lo.x, ("knee_lower", "shin_shell"), "lower")
    exact = {j: d.qpos[adr(j)] for j in MIMIC}
    R = d.xmat[foot].reshape(3, 3)
    pitch = np.arctan2(R[0, 2], R[2, 2]) - np.arctan2(R_ideal[0, 2], R_ideal[2, 2])
    resid = max(np.abs(up.fun).max(), np.abs(lo.fun).max())
    return {j: exact[j] - ideal[j] for j in MIMIC}, np.linalg.norm(d.xpos[foot] - sole_ideal), pitch, resid


def closed_chain_model(fixed_base=False):
    """The robot with mimic couplings replaced by true loop-closing `connect` constraints."""
    xml = open(os.path.join(MJCF_DIR, "zang.xml")).read()
    loops = sorted(set(re.findall(r'site name="(\w+_loop)_a"', xml)))
    connects = "".join(f'\n    <connect site1="{l}_a" site2="{l}_b" solref="0.002 1"/>' for l in loops)
    xml = re.sub(r"<equality>.*?</equality>", f"<equality>{connects}\n  </equality>", xml, flags=re.S)
    xml = xml.replace("<worldbody>", '<worldbody>\n    <geom name="floor" type="plane" size="0 0 0.05"/>', 1)
    if fixed_base:
        xml = xml.replace("<freejoint/>", "", 1)
        xml = re.sub(r'(<key name="home" qpos=")(\S+ ){7}', r"\1", xml)  # drop the 7 free-joint coordinates
    cwd = os.getcwd()
    os.chdir(MJCF_DIR)  # mesh paths are relative to the MJCF folder
    try:
        return mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(cwd)


def sweep(m, lo, hi):
    worst = dict(angle=0.0, sole=0.0, pitch=0.0, resid=0.0)
    for side in ("left", "right"):
        for qh in np.linspace(lo, hi, 13):
            for qk in np.linspace(lo, hi, 13):
                dq, ds, dp, res = kinematic_check(m, side, qh, qk)
                worst["angle"] = max(worst["angle"], max(abs(v) for v in dq.values()))
                worst["sole"], worst["pitch"] = max(worst["sole"], ds), max(worst["pitch"], abs(dp))
                worst["resid"] = max(worst["resid"], res)
    print(f"   hip/knee pitch in [{lo}, {hi}] rad: loops closed to {worst['resid']*1e6:.2f} um; pin angles off by "
          f"<= {np.degrees(worst['angle']):.2f} deg; foot off by <= {worst['sole']*1000:.2f} mm and {np.degrees(worst['pitch']):.2f} deg pitch")


def pin_error(mc, d):
    err = 0.0
    for s in ("left", "right"):
        for j, (drv, mul) in MIMIC.items():
            q = d.qpos[mc.jnt_qposadr[mc.joint(f"{s}_{j}").id]]
            err = max(err, abs(q - mul * d.qpos[mc.jnt_qposadr[mc.joint(f"{s}_{drv}").id]]))
    gap = max(np.linalg.norm(d.site_xpos[i] - d.site_xpos[mc.site(mc.site(i).name[:-1] + "b").id])
              for i in range(mc.nsite) if mc.site(i).name.endswith("_a"))
    return err, gap


def run_poses(mc, poses, seconds=2.0):
    d = mujoco.MjData(mc)
    key = mc.key("home").id
    mujoco.mj_resetDataKeyframe(mc, d, key)
    act = {mc.actuator(i).name: i for i in range(mc.nu)}
    pelvis = mc.body("pelvis").id
    for name, target in poses.items():
        for _ in range(int(seconds / mc.opt.timestep)):
            d.ctrl[:] = mc.key_ctrl[key]
            for k, v in target.items():
                d.ctrl[act[k]] = v
            mujoco.mj_step(mc, d)
        err, gap = pin_error(mc, d)
        up = d.xmat[pelvis].reshape(3, 3)[:, 2]
        print(f"   {name:30s} pins vs mimic <= {np.degrees(err):.2f} deg, loop gap <= {gap*1000:.3f} mm, "
              f"pelvis tilt {np.degrees(np.arccos(min(1, up[2]))):.1f} deg")


def main():
    m = mujoco.MjModel.from_xml_path(os.path.join(MJCF_DIR, "scene.xml"))
    print("1) exact four-bar closure (CAD pin positions) vs ideal-parallelogram mimic, both legs:")
    sweep(m, -0.9, 0.3)
    sweep(m, -1.2, 0.6)
    both = lambda h, k: {f"{s}_hip_pitch": h for s in ("left", "right")} | {f"{s}_knee_pitch": k for s in ("left", "right")}
    print("2a) MuJoCo closed chain (pins free, loops closed by connect constraints), standing on the floor:")
    run_poses(closed_chain_model(), {"home": {}, "deep squat (-0.9, -0.9)": both(-0.9, -0.9)})
    print("2b) same, pelvis fixed in the air, arbitrary leg poses:")
    run_poses(closed_chain_model(fixed_base=True), {"leg forward (-0.9, -0.2)": both(-0.9, -0.2),
              "leg back (0.4, 0.0)": both(0.4, 0.0), "knee deep (-1.2, -1.2)": both(-1.2, -1.2),
              "mixed (-0.3, 0.5)": both(-0.3, 0.5)})


if __name__ == "__main__":
    main()
