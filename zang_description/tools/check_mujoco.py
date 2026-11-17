"""Compile the MJCF, check linkage loop closure, list self-contacts at home, hold home and squat deeper."""
import os, sys, numpy as np, mujoco
scene = os.path.join(os.path.dirname(__file__), "..", "mjcf", "scene.xml")
m = mujoco.MjModel.from_xml_path(scene)
d = mujoco.MjData(m)
print(f"bodies={m.nbody} joints={m.njnt} actuators={m.nu} equalities={m.neq} mass={mujoco.mj_getTotalmass(m):.3f}kg")
qadr = lambda j: m.jnt_qposadr[m.joint(j).id]
closures = sorted({m.site(i).name[:-2] for i in range(m.nsite) if m.site(i).name.endswith("_a")})

def closure_error():
    return max(np.linalg.norm(d.site(c + "_a").xpos - d.site(c + "_b").xpos) for c in closures)

def apply_mimic():
    for e in range(m.neq):
        if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
            d.qpos[m.jnt_qposadr[m.eq_obj1id[e]]] = m.eq_data[e][1] * d.qpos[m.jnt_qposadr[m.eq_obj2id[e]]]

worst = 0
for qh in np.linspace(-1.2, 1.2, 9):
    for qk in np.linspace(-1.2, 1.2, 9):
        d.qpos[:] = m.qpos0
        for s in ("left", "right"):
            d.qpos[qadr(f"{s}_hip_pitch")], d.qpos[qadr(f"{s}_knee_pitch")] = qh, qk
        apply_mimic()
        mujoco.mj_kinematics(m, d)
        worst = max(worst, closure_error())
print(f"kinematic loop-closure error over hip/knee sweep: {worst*1000:.2f} mm")

key = m.key("home").id
mujoco.mj_resetDataKeyframe(m, d, key)
mujoco.mj_forward(m, d)
bn = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
pairs = {tuple(sorted((bn(c.geom1) or "world", bn(c.geom2) or "world"))) for c in d.contact[:d.ncon] if c.dist < 0}
print("initial penetrations:", pairs or "none")

pelvis = m.body("pelvis").id
act = {m.actuator(i).name: i for i in range(m.nu)}
def run(seconds, target):
    for _ in range(int(seconds / m.opt.timestep)):
        d.ctrl[:] = m.key_ctrl[key]
        for k, v in target(d.time).items():
            d.ctrl[act[k]] = v
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            sys.exit("simulation diverged")
def report(tag):
    up = d.xmat[pelvis].reshape(3, 3)[:, 2]
    print(f"{tag}: pelvis z {d.xpos[pelvis][2]*1000:.1f} mm, tilt {np.degrees(np.arccos(np.clip(up[2], -1, 1))):.1f} deg, "
          f"loop error {closure_error()*1000:.2f} mm, contacts {d.ncon}")
z0 = d.xpos[pelvis][2]
print(f"start (home keyframe): pelvis z {z0*1000:.1f} mm")
run(3.0, lambda t: {})
report("hold home 3 s")
squat = lambda t: {f"{s}_{j}": -0.5 - 0.4 * min(t / 2.0, 1.0) for s in ("left", "right") for j in ("hip_pitch", "knee_pitch")}
run(4.0, lambda t: squat(t - 3.0) if t > 3.0 else {})
report("deep squat (hip=knee=-0.9)")
