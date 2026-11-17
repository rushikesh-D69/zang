"""RL-side Zang model: loads the canonical MJCF and adds what RL needs, without editing the source files.

Source of truth:
  * geometry, masses, joints, equality loops: zang_description/mjcf/scene.xml (generated from the STEP)
  * joint order on the wire:                  ai_stack/config/robot.json (index order)

Added here, programmatically (mujoco.MjSpec):
  * an IMU site (default on the chest, where the P4 board is expected; TBD on the real robot) with gyro,
    accelerometer and framequat sensors;
  * a sole site per foot (centre of the sole, from the collision meshes) for contact / clearance / slip;
  * per-actuator position sensors (what the servo bus reads) and actuator-force sensors (for energy terms);
  * pelvis framelinvel / framepos (privileged, critic only);
  * foot collision: `feet="box"` (default) replaces the three convex foot meshes per foot with one box that
    matches the measured sole rectangle (13 x 8.5 cm, bottom at the sole). Box-plane contact is exact, cheap,
    and supported by MJX (the GPU backend), so CPU and GPU train on identical contact geometry; sim2sim.py
    checks transfer back to the real meshes (`feet="mesh"`).
  * collision filtering: only the feet and the floor collide by default (mesh-mesh self collision is the
    single biggest cost and the stand/walk tasks terminate long before limbs hit each other). Set
    `collisions="legs"` to also let the two legs collide (needed once gaits get narrow; CPU backends only).

The closed parallelogram loops are kept exactly as in the source MJCF (joint-coupling equalities, `mjEQ_JOINT`),
and `loop_gap()` measures the true geometric loop closure (distance between the paired `*_loop_a/b` sites) so
rollouts can verify the loops hold.
"""
from __future__ import annotations

import dataclasses
import json
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RL_ROOT = os.path.dirname(HERE)
AI_STACK = os.path.dirname(RL_ROOT)
ZANG = os.path.dirname(AI_STACK)
ROBOT_JSON = os.path.join(AI_STACK, "config", "robot.json")
SCENE_XML = os.path.join(ZANG, "zang_description", "mjcf", "scene.xml")

# Sole rectangle in the foot frame, measured from the collision meshes at the home keyframe
# (see README "Model facts"). Left foot y range is [-0.022, 0.0626]; the right foot mirrors it.
SOLE_Z = -0.0236
FOOT_BOX_HALF_Z = 0.004       # box-foot thickness/2; the box bottom is the sole plane
SOLE_X = (-0.040, 0.090)
SOLE_Y_LEFT = (-0.022, 0.0626)

LOOP_PAIRS = [(f"{s}_{lvl}_loop_a", f"{s}_{lvl}_loop_b") for s in ("left", "right") for lvl in ("upper", "lower")]


def load_joint_table(path: str = ROBOT_JSON) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    joints = sorted(cfg["joints"], key=lambda j: j["index"])
    assert [j["index"] for j in joints] == list(range(cfg["n_joints"])), "robot.json indices must be 0..n-1"
    return cfg | {"joints": joints}


@dataclasses.dataclass
class ZangModel:
    """Compiled RL model plus the index maps every other module needs."""

    model: mujoco.MjModel
    joint_names: list[str]          # wire order (robot.json)
    act_ids: np.ndarray             # actuator id for wire index i
    qpos_adr: np.ndarray            # qpos address of the servo joint for wire index i
    dof_adr: np.ndarray             # dof address of the servo joint for wire index i
    home_q: np.ndarray              # home joint angles, wire order (rad)
    lower: np.ndarray               # joint range, wire order
    upper: np.ndarray
    home_qpos: np.ndarray           # full qpos of the home keyframe
    sensor_slices: dict             # sensor name -> slice into sensordata
    imu_body: str
    foot_site_ids: tuple[int, int]
    loop_site_ids: list[tuple[int, int]]
    loop_gap_home: np.ndarray       # loop-site distances at home (non-zero: CAD pin spacing is not ideal)
    eq_pairs: list[tuple[int, int, float]]   # (passive qpos adr, driving qpos adr, multiplier)

    @property
    def nj(self) -> int:
        return len(self.joint_names)

    def sensor(self, sensordata: np.ndarray, name: str) -> np.ndarray:
        return sensordata[..., self.sensor_slices[name]]


def build(imu_body: str = "chest", collisions: str = "feet", scene: str = SCENE_XML,
          robot_json: str = ROBOT_JSON, timestep: float | None = None,
          integrator: int | None = None, strip_visuals: bool = True, feet: str = "box") -> ZangModel:
    table = load_joint_table(robot_json)
    names = [j["name"] for j in table["joints"]]

    spec = mujoco.MjSpec.from_file(scene)

    # ---- sites ----------------------------------------------------------------------------------------------
    spec.body(imu_body).add_site(name="imu", pos=[0, 0, 0], size=[0.004] * 3, group=4)
    for side in ("left", "right"):
        y0, y1 = SOLE_Y_LEFT if side == "left" else (-SOLE_Y_LEFT[1], -SOLE_Y_LEFT[0])
        cx, cy = 0.5 * sum(SOLE_X), 0.5 * (y0 + y1)
        spec.body(f"{side}_foot").add_site(name=f"{side}_sole", pos=[cx, cy, SOLE_Z],
                                           type=mujoco.mjtGeom.mjGEOM_BOX,
                                           size=[0.5 * (SOLE_X[1] - SOLE_X[0]), 0.5 * (y1 - y0), 0.002], group=4)

    # ---- sensors ------------------------------------------------------------------------------------------------
    S, O = mujoco.mjtSensor, mujoco.mjtObj
    add = lambda name, typ, objtype, obj, **kw: spec.add_sensor(name=name, type=typ, objtype=objtype, objname=obj, **kw)
    add("imu_gyro", S.mjSENS_GYRO, O.mjOBJ_SITE, "imu")
    add("imu_acc", S.mjSENS_ACCELEROMETER, O.mjOBJ_SITE, "imu")
    add("imu_quat", S.mjSENS_FRAMEQUAT, O.mjOBJ_SITE, "imu")
    add("pelvis_pos", S.mjSENS_FRAMEPOS, O.mjOBJ_XBODY, "pelvis")
    add("pelvis_linvel", S.mjSENS_FRAMELINVEL, O.mjOBJ_XBODY, "pelvis")
    add("pelvis_quat", S.mjSENS_FRAMEQUAT, O.mjOBJ_XBODY, "pelvis")
    for side in ("left", "right"):
        add(f"{side}_sole_pos", S.mjSENS_FRAMEPOS, O.mjOBJ_SITE, f"{side}_sole")
        add(f"{side}_sole_linvel", S.mjSENS_FRAMELINVEL, O.mjOBJ_SITE, f"{side}_sole")
        add(f"{side}_sole_zaxis", S.mjSENS_FRAMEZAXIS, O.mjOBJ_SITE, f"{side}_sole")
    for n in names:
        add(f"q_{n}", S.mjSENS_ACTUATORPOS, O.mjOBJ_ACTUATOR, n)
    for n in names:
        add(f"dq_{n}", S.mjSENS_ACTUATORVEL, O.mjOBJ_ACTUATOR, n)
    for n in names:
        add(f"tau_{n}", S.mjSENS_ACTUATORFRC, O.mjOBJ_ACTUATOR, n)
    for a, b in LOOP_PAIRS:
        add(f"site_{a}", S.mjSENS_FRAMEPOS, O.mjOBJ_SITE, a)
        add(f"site_{b}", S.mjSENS_FRAMEPOS, O.mjOBJ_SITE, b)

    # ---- foot collision geometry ------------------------------------------------------------------------------
    if feet == "box":
        for side in ("left", "right"):
            body = spec.body(f"{side}_foot")
            for g in list(spec.geoms):
                if g.parent is not None and g.parent.name == body.name and (g.contype or g.conaffinity):
                    spec.delete(g)
            y0, y1 = SOLE_Y_LEFT if side == "left" else (-SOLE_Y_LEFT[1], -SOLE_Y_LEFT[0])
            body.add_geom(name=f"{side}_sole_box", type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[0.5 * (SOLE_X[1] - SOLE_X[0]), 0.5 * (y1 - y0), FOOT_BOX_HALF_Z],
                          pos=[0.5 * sum(SOLE_X), 0.5 * (y0 + y1), SOLE_Z + FOOT_BOX_HALF_Z],
                          friction=[0.9, 0.01, 0.001], density=0, group=3, contype=1, conaffinity=1)
    elif feet != "mesh":
        raise ValueError(f"feet must be 'box' or 'mesh', got {feet!r}")

    # ---- collision filtering ------------------------------------------------------------------------------------
    keep_bodies = {"left_foot", "right_foot"}
    if collisions == "legs":
        keep_bodies |= {f"{s}_{p}" for s in ("left", "right")
                        for p in ("foot", "ankle", "shin", "shin_shell", "knee", "thigh", "thigh_shell")}
    elif collisions == "all":
        keep_bodies = None
    if keep_bodies is not None:
        for g in spec.geoms:
            if g.contype == 0 and g.conaffinity == 0:
                continue
            body = g.parent.name if g.parent is not None else ""
            if body == "world" or g.name == "floor":
                continue
            if body not in keep_bodies:
                g.contype, g.conaffinity = 0, 0
            elif collisions == "feet":
                g.contype, g.conaffinity = 1, 0      # feet hit the floor (conaffinity 1) only

    if strip_visuals:
        # Non-colliding geoms are only for rendering (all geoms have density 0; masses come from the explicit
        # inertials), and their meshes dominate MjModel size (~75 MB per copy). Training keeps one model per world.
        for g in list(spec.geoms):
            if g.contype == 0 and g.conaffinity == 0:
                spec.delete(g)
        used = {g.meshname for g in spec.geoms}
        for msh in list(spec.meshes):
            if msh.name not in used:
                spec.delete(msh)

    if timestep is not None:
        spec.option.timestep = timestep
    if integrator is not None:
        spec.option.integrator = integrator
    model = spec.compile()
    return _index(model, names, imu_body)


def _index(model: mujoco.MjModel, names: list[str], imu_body: str) -> ZangModel:
    act_ids = np.array([model.actuator(n).id for n in names])
    jids = np.array([model.joint(n).id for n in names])
    key = model.key("home").id
    home_qpos = model.key_qpos[key].copy()
    qadr, dadr = model.jnt_qposadr[jids], model.jnt_dofadr[jids]

    slices, adr = {}, 0
    for i in range(model.nsensor):
        dim = int(model.sensor_dim[i])
        slices[model.sensor(i).name] = slice(adr, adr + dim)
        adr += dim

    eq_pairs = []
    for e in range(model.neq):
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
            j1, j2 = model.eq_obj1id[e], model.eq_obj2id[e]
            eq_pairs.append((int(model.jnt_qposadr[j1]), int(model.jnt_qposadr[j2]), float(model.eq_data[e, 1])))

    zm = ZangModel(
        model=model, joint_names=names, act_ids=act_ids, qpos_adr=qadr, dof_adr=dadr,
        home_q=home_qpos[qadr].copy(), lower=model.jnt_range[jids, 0].copy(), upper=model.jnt_range[jids, 1].copy(),
        home_qpos=home_qpos, sensor_slices=slices, imu_body=imu_body,
        foot_site_ids=(model.site("left_sole").id, model.site("right_sole").id),
        loop_site_ids=[(model.site(a).id, model.site(b).id) for a, b in LOOP_PAIRS],
        loop_gap_home=np.zeros(len(LOOP_PAIRS)), eq_pairs=eq_pairs,
    )
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, key)
    mujoco.mj_forward(model, d)
    zm.loop_gap_home = loop_gap(zm, d.sensordata)
    return zm


def loop_gap(zm: ZangModel, sensordata: np.ndarray) -> np.ndarray:
    """Distance between the paired parallelogram pin sites (m). Shape (..., 4): left upper, left lower, right ..."""
    out = []
    for a, b in LOOP_PAIRS:
        out.append(np.linalg.norm(zm.sensor(sensordata, f"site_{a}") - zm.sensor(sensordata, f"site_{b}"), axis=-1))
    return np.stack(out, -1)


def eq_residual(zm: ZangModel, qpos: np.ndarray) -> np.ndarray:
    """Joint-coupling equality violation |q_passive - k * q_servo| (rad), shape (..., n_eq)."""
    return np.stack([np.abs(qpos[..., p] - k * qpos[..., s]) for p, s, k in zm.eq_pairs], -1)


if __name__ == "__main__":
    zm = build()
    m = zm.model
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} neq={m.neq} nsensordata={m.nsensordata} mass={m.body_subtreemass[1]:.3f} kg")
    print("wire order:", zm.joint_names)
    print("actuator ids:", zm.act_ids.tolist())
    print("home q:", np.round(zm.home_q, 3).tolist())
    print("loop gap at home (mm):", np.round(zm.loop_gap_home * 1e3, 3).tolist())
