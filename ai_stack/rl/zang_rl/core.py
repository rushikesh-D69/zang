"""Task definition shared by EVERY Zang env: domain randomisation, initial state, servo command path,
observations, rewards, terminations, commands and pushes.

One implementation, three front ends:
  * zang_rl/env.py      rsl_rl VecEnv, many worlds on the CPU via mujoco.rollout       (xp = numpy)
  * zang_rl/gym_env.py  Gymnasium MujocoEnv ZangStand-v0 / ZangWalk-v0, one world      (xp = numpy)
  * zang_rl/mjx_env.py  MJX (JAX) env for brax PPO on a GPU, thousands of worlds        (xp = jax.numpy)

Every function takes the array module `xp` (numpy or jax.numpy) and is pure math. Randomness always enters as
arrays of unit uniforms `u` (in [0, 1)) or standard normals `n` drawn by the caller with its own RNG (numpy
Generator or jax.random), so the distributions are identical by construction. Arrays are batch-first (B worlds)
unless a function says "single world". tests/test_backend_parity.py checks numpy vs JAX on identical inputs.
"""
from __future__ import annotations

import numpy as np

from . import obs as O
from .config import EnvCfg
from .robot import ZangModel

SOLE_CLEARANCE = 0.0008          # initial height of the lower sole above the floor (m)
CONTACT_Z = 0.006                # sole site below this height = foot in contact (m)


def _set(xp, arr, idx, val):
    """arr[idx] = val for numpy (on a copy) and jax (functional)."""
    if xp is np:
        out = np.array(arr, copy=True)
        out[idx] = val
        return out
    return arr.at[idx].set(val)


def _lerp(r, u):
    return r[0] + (r[1] - r[0]) * u


# ============================================================================================ model bookkeeping
def capture_base(m) -> dict:
    """Nominal parameters that domain randomisation scales (from the compiled MjModel, before any randomisation)."""
    foot = np.array([g for g in range(m.ngeom) if m.geom_contype[g]
                     and m.body(m.geom_bodyid[g]).name in ("left_foot", "right_foot")])
    return dict(mass=m.body_mass.copy(), inertia=m.body_inertia.copy(), ipos=m.body_ipos.copy(),
                friction=m.geom_friction.copy(), eq=m.eq_data.copy(), foot_geoms=foot,
                com_bodies=np.array([m.body("pelvis").id, m.body("chest").id]),
                gainprm=m.actuator_gainprm.copy(), biasprm=m.actuator_biasprm.copy(),
                forcerange=m.actuator_forcerange.copy(), ctrlrange=m.actuator_ctrlrange.copy(),
                damping=m.dof_damping.copy(), armature=m.dof_armature.copy(), frictionloss=m.dof_frictionloss.copy(),
                nbody=m.nbody, neq=m.neq, timestep=float(m.opt.timestep))


DR_FIELDS = ("body_mass", "body_inertia", "body_ipos", "geom_friction", "actuator_gainprm", "actuator_biasprm",
             "actuator_forcerange", "actuator_ctrlrange", "dof_damping", "dof_armature", "dof_frictionloss", "eq_data")


def dr_num_uniforms(base: dict, nj: int) -> int:
    return 1 + base["nbody"] + 6 + 1 + 5 * nj + base["neq"] + nj + 1


def dr_from_uniforms(xp, base: dict, zm: ZangModel, cfg: EnvCfg, u):
    """Single world. u: (dr_num_uniforms,) unit uniforms. Returns (model fields dict, calib (nj,), latency
    substeps (int), summary (4,): total mass scale, mean kp scale, foot friction, latency s)."""
    sv, dr, nj = cfg.servo, cfg.dr, zm.nj
    on = 1.0 if dr.enabled else 0.0
    i = 0

    def take(n):
        nonlocal i
        out = u[i:i + n]
        i += n
        return out

    def scale(r, n):                 # randomised scale, or 1 when DR is off
        return on * _lerp(r, take(n)) + (1 - on)

    tot = scale(dr.total_mass_scale, 1)[0]
    per = scale(dr.body_mass_scale, base["nbody"])
    per = _set(xp, per, 0, 1.0)
    com = on * (2 * take(6) - 1) * dr.com_shift
    fr = on * _lerp(dr.foot_friction, take(1)[0]) + (1 - on) * float(base["friction"][base["foot_geoms"][0], 0])
    kp = sv.kp * scale(dr.kp_scale, nj)
    fl = sv.force_limit * scale(dr.force_scale, nj)
    damp = sv.damping * scale(dr.damping_scale, nj)
    arm = sv.armature * scale(dr.armature_scale, nj)
    fric = sv.frictionloss * scale(dr.frictionloss_scale, nj)
    coup = 1.0 + on * (2 * take(base["neq"]) - 1) * dr.loop_coupling
    calib = on * (2 * take(nj) - 1) * dr.calib_offset
    lat = on * _lerp(dr.latency_s, take(1)[0])

    a, d = zm.act_ids, zm.dof_adr
    ipos = _set(xp, xp.asarray(base["ipos"]), base["com_bodies"], xp.asarray(base["ipos"])[base["com_bodies"]] + com.reshape(2, 3))
    fields = dict(
        body_mass=xp.asarray(base["mass"]) * per * tot,
        body_inertia=xp.asarray(base["inertia"]) * (per * tot)[:, None],
        body_ipos=ipos,
        geom_friction=_set(xp, xp.asarray(base["friction"]), (base["foot_geoms"], 0), fr),
        actuator_gainprm=_set(xp, xp.asarray(base["gainprm"]), (a, 0), kp),
        actuator_biasprm=_set(xp, xp.asarray(base["biasprm"]), (a, 1), -kp),
        actuator_forcerange=_set(xp, xp.asarray(base["forcerange"]), (a, slice(None)), xp.stack([-fl, fl], -1)),
        actuator_ctrlrange=_set(xp, xp.asarray(base["ctrlrange"]), (a, slice(None)),
                                xp.stack([xp.asarray(zm.lower) - 0.2, xp.asarray(zm.upper) + 0.2], -1)),
        dof_damping=_set(xp, xp.asarray(base["damping"]), d, damp),
        dof_armature=_set(xp, xp.asarray(base["armature"]), d, arm),
        dof_frictionloss=_set(xp, xp.asarray(base["frictionloss"]), d, fric),
        eq_data=_set(xp, xp.asarray(base["eq"]), (slice(None), 1), xp.asarray(base["eq"])[:, 1] * coup),
    )
    lat_sub = xp.round(lat / base["timestep"]).astype(xp.int32)
    summary = xp.stack([tot, kp.mean() / sv.kp, fr, lat])
    return fields, calib, lat_sub, summary


def apply_fields_numpy(m, fields: dict):
    """Write randomised fields into a mujoco.MjModel in place (CPU backends)."""
    for k, v in fields.items():
        getattr(m, k)[:] = v


# ================================================================================================ initial state
def init_num_uniforms(zm: ZangModel) -> int:
    return zm.nj + 1


def init_qpos_from_uniforms(xp, zm: ZangModel, cfg: EnvCfg, u):
    """Single world: home crouch + joint noise + random heading, parallelogram pins consistent.
    The caller then lifts the robot so the lower sole is SOLE_CLEARANCE above the floor (needs kinematics)."""
    q = xp.asarray(zm.home_qpos, dtype=xp.float64 if xp is np else None)
    q = _set(xp, q, zm.qpos_adr, xp.asarray(zm.home_qpos)[zm.qpos_adr] + (2 * u[:zm.nj] - 1) * cfg.init_noise_q)
    for p, s, k in zm.eq_pairs:
        q = _set(xp, q, p, k * q[s])
    yaw = (2 * u[zm.nj] - 1) * cfg.init_yaw
    return _set(xp, q, slice(3, 7), xp.stack([xp.cos(yaw / 2), 0 * yaw, 0 * yaw, xp.sin(yaw / 2)]))


# ================================================================================================ sensors setup
def sensor_episode_from_uniforms(xp, cfg: EnvCfg, u, nsub: int, dt: float):
    """Batch: u (B, 7). Per-episode gyro bias (B,3), IMU mounting rotation (B,3,3), bus read delay (B,) substeps."""
    nz = cfg.noise
    on = 1.0 if nz.enabled else 0.0
    bias = on * (2 * u[:, 0:3] - 1) * nz.gyro_bias
    ang = on * (2 * u[:, 3:6] - 1) * nz.imu_tilt
    delay = xp.clip(xp.round(on * _lerp(nz.q_delay_s, u[:, 6]) / dt), 0, 2 * nsub - 1).astype(xp.int32)
    return bias, _rot_zyx(xp, ang), delay


def _rot_zyx(xp, a):
    rx, ry, rz = a[:, 0], a[:, 1], a[:, 2]
    cx, sx, cy, sy, cz, sz = xp.cos(rx), xp.sin(rx), xp.cos(ry), xp.sin(ry), xp.cos(rz), xp.sin(rz)
    # Rz @ Ry @ Rx
    return xp.stack([
        xp.stack([cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx], -1),
        xp.stack([sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], -1),
        xp.stack([-sy, cy * sx, cy * cx], -1)], -2)


# ================================================================================================ command path
def action_to_target(xp, zm: ZangModel, cfg: EnvCfg, action):
    return xp.clip(xp.asarray(zm.home_q) + cfg.action_scale * xp.clip(action, -10.0, 10.0), zm.lower, zm.upper)


def servo_profile(xp, target_prev, target, lat_sub, nsub: int, dt: float, servo):
    """Per-substep joint targets (B, nsub, nj): latency, DS ramp over the period (or step), speed cap."""
    k = xp.arange(nsub)[None, :, None]
    L = xp.reshape(lat_sub, (-1, 1, 1))
    if servo.interp == "ramp":
        frac = xp.clip((k - L + 1) / nsub, 0.0, 1.0)
    else:
        frac = (k >= L).astype(target.dtype)
    cap = servo.vmax * dt * xp.clip(k - L + 1, 0, None)
    return target_prev[:, None, :] + xp.clip((target - target_prev)[:, None, :] * frac, -cap, cap)


# ================================================================================================ observations
def q_sensor_slice(zm: ZangModel, kind: str = "q") -> slice:
    s0 = zm.sensor_slices[f"{kind}_{zm.joint_names[0]}"].start
    return slice(s0, s0 + zm.nj)


def cmd_phase(xp, cfg: EnvCfg, command, phase):
    if cfg.task == "stand":
        return xp.zeros_like(command), xp.zeros((command.shape[0], 2), dtype=command.dtype)
    moving = (xp.linalg.norm(command, axis=-1, keepdims=True) > 1e-6).astype(command.dtype)
    return command, xp.stack([xp.sin(phase), xp.cos(phase)], -1) * moving


def observe(xp, zm: ZangModel, cfg: EnvCfg, sens, q_hist, qdelay_sub, calib, q_read_prev, imu_R, gyro_bias,
            last_action, command, phase, dr_summary, noise=None):
    """Returns (actor obs, critic obs, q_read). sens = last-substep sensordata (B, nsensor); q_hist = true joint
    angles per substep over the last two policy steps (B, 2*nsub, nj). noise = dict of standard normals
    {"q": (B,nj), "gyro": (B,3), "grav": (B,3)} or None (no white noise)."""
    nz = cfg.noise
    B = sens.shape[0]
    gyro_true = zm.sensor(sens, "imu_gyro")
    grav_true = O.gravity_from_quat(zm.sensor(sens, "imu_quat"), xp)
    q_true = sens[:, q_sensor_slice(zm, "q")]
    dq_true = sens[:, q_sensor_slice(zm, "dq")]
    q_read = q_hist[xp.arange(B), q_hist.shape[1] - 1 - qdelay_sub] - calib
    gyro = xp.einsum("bij,bj->bi", imu_R, gyro_true) + gyro_bias
    grav = xp.einsum("bij,bj->bi", imu_R, grav_true)
    if nz.enabled:
        if noise is not None:
            q_read = q_read + nz.q_std * noise["q"]
            gyro = gyro + nz.gyro_std * noise["gyro"]
            grav = grav + nz.gravity_std * noise["grav"]
        q_read = xp.round(q_read / nz.q_quant) * nz.q_quant
    grav = grav / xp.linalg.norm(grav, axis=-1, keepdims=True)
    dq_est = (q_read - q_read_prev) / cfg.ctrl_dt
    cmd, ph = cmd_phase(xp, cfg, command, phase)
    home = xp.asarray(zm.home_q)
    imu = (gyro, grav) if cfg.imu else (None, None)
    actor = O.build_actor_obs(q_read - home, dq_est, last_action, cmd, ph, *imu, xp=xp)
    # critic (training only) always gets true IMU signals: privileged, asymmetric actor-critic
    clean = O.build_actor_obs(q_true - home, dq_true, last_action, cmd, ph, gyro_true, grav_true, xp=xp)
    yaw = O.yaw_from_quat(zm.sensor(sens, "pelvis_quat"), xp)
    v_head = rotate_inv_yaw(xp, zm.sensor(sens, "pelvis_linvel"), yaw)
    feet_z = xp.stack([zm.sensor(sens, "left_sole_pos")[:, 2], zm.sensor(sens, "right_sole_pos")[:, 2]], -1)
    priv = xp.concatenate([v_head, zm.sensor(sens, "pelvis_pos")[:, 2:3], (feet_z < CONTACT_Z).astype(feet_z.dtype),
                           feet_z, dr_summary], -1).astype(xp.float32)
    return actor, xp.concatenate([clean, priv], -1), q_read


def critic_dim() -> int:
    return O.obs_dim(True) + 12


def rotate_inv_yaw(xp, vec_world, yaw):
    c, s = xp.cos(yaw), xp.sin(yaw)
    x, y = vec_world[..., 0], vec_world[..., 1]
    return xp.stack([c * x + s * y, -s * x + c * y, vec_world[..., 2]], -1)


# ===================================================================================================== rewards
def rewards(xp, zm: ZangModel, cfg: EnvCfg, sens, finite, last_action, prev_action, command, phase):
    """Returns (total reward (B,), per-term dict scaled by weight * ctrl_dt, terminated (B,) bool)."""
    B = sens.shape[0]
    f = sens.dtype
    pel_quat = zm.sensor(sens, "pelvis_quat")
    g = O.gravity_from_quat(pel_quat, xp)
    tilt = xp.arccos(xp.clip(-g[:, 2], -1, 1))
    z = zm.sensor(sens, "pelvis_pos")[:, 2]
    v = rotate_inv_yaw(xp, zm.sensor(sens, "pelvis_linvel"), O.yaw_from_quat(pel_quat, xp))
    w = zm.sensor(sens, "imu_gyro")
    q = sens[:, q_sensor_slice(zm, "q")]
    tau = sens[:, q_sensor_slice(zm, "tau")]
    feet_pos = xp.stack([zm.sensor(sens, "left_sole_pos"), zm.sensor(sens, "right_sole_pos")], 1)
    feet_vel = xp.stack([zm.sensor(sens, "left_sole_linvel"), zm.sensor(sens, "right_sole_linvel")], 1)
    contact = feet_pos[:, :, 2] < CONTACT_Z
    h0 = float(zm.home_qpos[2])
    lo, hi, home = xp.asarray(zm.lower), xp.asarray(zm.upper), xp.asarray(zm.home_q)
    terminated = (tilt > cfg.max_tilt) | (z < cfg.min_height) | ~finite
    r = {
        "alive": xp.ones(B, dtype=f),
        "upright": xp.exp(-(tilt ** 2) / 0.05),
        "height": xp.exp(-((z - h0) ** 2) / 0.02 ** 2),
        "pose": xp.exp(-xp.sum((q - home) ** 2, -1) / 0.5),
        "still": xp.exp(-(xp.sum(v[:, :2] ** 2, -1) / 0.01 + xp.sum(w ** 2, -1) / 1.0)),
        "feet_contact": xp.all(contact, -1).astype(f),
        "foot_slip": xp.sum(xp.sum(feet_vel[:, :, :2] ** 2, -1) * contact, -1),
        "action_rate": xp.sum((last_action - prev_action) ** 2, -1),
        "torque": xp.sum(tau ** 2, -1),
        "joint_limit": xp.sum(xp.clip(lo + 0.1 - q, 0, None) + xp.clip(q - hi + 0.1, 0, None), -1),
        "termination": terminated.astype(f),
    }
    if cfg.task == "joystick":
        r["track_lin_vel"] = xp.exp(-xp.sum((command[:, :2] - v[:, :2]) ** 2, -1) / 0.01)
        r["track_ang_vel"] = xp.exp(-((command[:, 2] - w[:, 2]) ** 2) / 0.1)
        moving = xp.linalg.norm(command, axis=-1) > 1e-6
        stance_l = xp.sin(phase) >= 0
        want = xp.stack([stance_l, ~stance_l], -1) | ~moving[:, None]
        r["gait_contact"] = xp.mean((contact == want).astype(f), -1)
        swing_target = cfg.swing_height * xp.abs(xp.sin(phase))
        fz = feet_pos[:, :, 2]
        r["swing_height"] = xp.exp(-xp.sum(xp.where(want, fz, xp.abs(fz - swing_target[:, None])) ** 2, -1) / 0.01 ** 2)
        r["lin_vel_z"] = v[:, 2] ** 2
        r["ang_vel_xy"] = xp.sum(w[:, :2] ** 2, -1)
    terms = {k: cfg.reward_scales[k] * r[k] * cfg.ctrl_dt for k in cfg.reward_scales}
    total = sum(terms.values())
    return xp.nan_to_num(total, nan=-1.0), terms, terminated


# ========================================================================================== commands and pushes
def commands_from_uniforms(xp, cfg: EnvCfg, u):
    """Batch: u (B, 5) -> command (B, 3), resample timer (B,)."""
    c = cfg.command
    cmd = xp.stack([_lerp(c.vx, u[:, 0]), _lerp(c.vy, u[:, 1]), _lerp(c.wz, u[:, 2])], -1)
    cmd = xp.where((u[:, 3] < c.zero_prob)[:, None], 0.0, cmd)
    return cmd, _lerp(c.resample_s, u[:, 4])


def push_from_uniforms(xp, cfg: EnvCfg, u):
    """Batch: u (B, 6) -> qvel kick (B, 6) for the free joint (linear world xy, angular local), next timer (B,)."""
    p = cfg.push
    ang = 2 * np.pi * u[:, 0]
    mag = p.max_vel_xy * u[:, 1]
    zero = 0 * mag
    dv = xp.stack([mag * xp.cos(ang), mag * xp.sin(ang), zero,
                   (2 * u[:, 2] - 1) * p.max_ang_vel, (2 * u[:, 3] - 1) * p.max_ang_vel,
                   (2 * u[:, 4] - 1) * p.max_ang_vel], -1)
    return dv, _lerp(p.interval_s, u[:, 5])


def advance_phase(xp, cfg: EnvCfg, phase):
    return (phase + 2 * np.pi * cfg.ctrl_dt / cfg.gait_period_s) % (2 * np.pi)


# ================================================================================================ CPU helpers
def reset_world_numpy(m, d, base: dict, zm: ZangModel, cfg: EnvCfg, rng):
    """CPU backends: resample one world's parameters into MjModel m and put MjData d in its initial state.
    (No mj_setConst after randomisation: MJX cannot do it either, and both backends must match.)"""
    import mujoco
    fields, calib, lat_sub, summary = dr_from_uniforms(np, base, zm, cfg, rng.uniform(size=dr_num_uniforms(base, zm.nj)))
    apply_fields_numpy(m, fields)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = init_qpos_from_uniforms(np, zm, cfg, rng.uniform(size=init_num_uniforms(zm)))
    mujoco.mj_kinematics(m, d)
    d.qpos[2] += SOLE_CLEARANCE - min(d.site_xpos[s][2] for s in zm.foot_site_ids)
    d.ctrl[zm.act_ids] = d.qpos[zm.qpos_adr] + calib
    mujoco.mj_forward(m, d)
    return calib, int(lat_sub), summary


def normals_numpy(rng, B: int, nj: int) -> dict:
    return {"q": rng.normal(size=(B, nj)), "gyro": rng.normal(size=(B, 3)), "grav": rng.normal(size=(B, 3))}
