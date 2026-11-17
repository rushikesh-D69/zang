"""Sim-to-sim check: run a policy in plain CPU MuJoCo with physics settings the policy never trained on.

This loop is written independently of zang_rl/env.py, the way the P4 firmware will run the policy:
read sensors -> build the actor observation (zang_rl/obs.py, IMU-free unless the policy was trained with one) -> MLP -> joint targets -> DS-style ramp.
It runs a list of physics configurations and prints one row of metrics per configuration:

  train   : dt 2 ms, implicitfast, feet-only collision   (what training uses, no randomisation)
  sim2sim : dt 1 ms, RK4,          full mesh collision   (different integrator, timestep and contact set)

Metrics: fall rate, mean time to fall, joint tracking RMS (commanded target vs measured joint, deg),
base tilt mean / max (deg), parallelogram equality residual max (deg), pin-site loop gap change max (mm) and, for a
walking policy, the velocity command error (|v_xy in the heading frame - command|, after 1 s) and the mean forward speed.

  .venv/bin/python sim2sim.py --policy logs/ZangStand/<run>/policy.onnx
  .venv/bin/python sim2sim.py --policy home          # zero action = hold the home pose (baseline)
  .venv/bin/python sim2sim.py --policy <walk run>/policy.onnx --command 0.1 0 0   # walking policy (policy_meta.json task=joystick)
"""
import argparse
import collections
import json
import os
import time

import mujoco
import numpy as np

from zang_rl import obs as O
from zang_rl import robot
from zang_rl.config import NoiseCfg, ServoCfg

CONFIGS = {
    "train": dict(timestep=0.002, integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST, collisions="feet"),
    "sim2sim": dict(timestep=0.001, integrator=mujoco.mjtIntegrator.mjINT_RK4, collisions="all"),
    "euler": dict(timestep=0.001, integrator=mujoco.mjtIntegrator.mjINT_EULER, collisions="legs"),
}


class Policy:
    def __init__(self, spec: str):
        self.home = spec == "home"
        if not self.home:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(spec)
            self.inp = self.sess.get_inputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.home:
            return np.zeros(15)
        return self.sess.run(None, {self.inp: obs[None].astype(np.float32)})[0][0].astype(np.float64)


def apply_servo(zm, sv: ServoCfg):
    m, a, d = zm.model, zm.act_ids, zm.dof_adr
    m.actuator_gainprm[a, 0] = sv.kp
    m.actuator_biasprm[a, 1] = -sv.kp
    m.actuator_forcerange[a, 0], m.actuator_forcerange[a, 1] = -sv.force_limit, sv.force_limit
    m.actuator_ctrlrange[a, 0], m.actuator_ctrlrange[a, 1] = zm.lower - 0.2, zm.upper + 0.2
    m.dof_damping[d], m.dof_armature[d], m.dof_frictionloss[d] = sv.damping, sv.armature, sv.frictionloss


def run_config(name, policy, a, sv, nz):
    c = CONFIGS[name]
    zm = robot.build(imu_body=a.imu_body, collisions=c["collisions"], timestep=c["timestep"], integrator=c["integrator"])
    apply_servo(zm, sv)
    m = zm.model
    d = mujoco.MjData(m)
    nsub = int(round(a.ctrl_dt / m.opt.timestep))
    lat = int(round(a.latency / m.opt.timestep))
    qdelay = int(round(a.q_delay / m.opt.timestep))
    rng = np.random.default_rng(a.seed)
    falls, fall_t, track_sq, track_n, tilts, eq_max, gap_max = 0, [], 0.0, 0, [], 0.0, 0.0
    vel_err, vx = [], []
    t0 = time.time()
    for ep in range(a.episodes):
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        mujoco.mj_kinematics(m, d)
        d.qpos[2] += 0.0008 - min(d.site_xpos[s][2] for s in zm.foot_site_ids)
        d.ctrl[zm.act_ids] = zm.home_q
        mujoco.mj_forward(m, d)
        q_buf = collections.deque([d.sensordata[_q(zm)].copy()] * (qdelay + 1), maxlen=qdelay + 1)
        bias = rng.uniform(-nz.gyro_bias, nz.gyro_bias, 3) if a.noise else np.zeros(3)
        last_a = np.zeros(15)
        phase = 0.0
        target_prev = zm.home_q.copy()
        q_prev_read = q_buf[0].copy()
        push_every = int(round(a.push_interval / a.ctrl_dt))
        fell, post_fall = False, 0
        for k in range(int(round(a.seconds / a.ctrl_dt))):
            s = d.sensordata
            q_read = q_buf[0].copy()
            gyro = zm.sensor(s, "imu_gyro").copy()
            grav = O.gravity_from_quat(zm.sensor(s, "imu_quat"))
            if a.noise:
                q_read = np.round((q_read + rng.normal(0, nz.q_std, 15)) / nz.q_quant) * nz.q_quant
                gyro += bias + rng.normal(0, nz.gyro_std, 3)
                grav = grav + rng.normal(0, nz.gravity_std, 3)
                grav /= np.linalg.norm(grav)
            dq = (q_read - q_prev_read) / a.ctrl_dt
            q_prev_read = q_read
            imu = (gyro, grav) if a.imu else (None, None)
            if a.walk and np.linalg.norm(a.command) > 1e-6:        # same as core.cmd_phase: command + gait clock
                cmd, ph = np.asarray(a.command, float), np.array([np.sin(phase), np.cos(phase)])
            else:
                cmd, ph = np.zeros(3), np.zeros(2)
            phase = (phase + 2 * np.pi * a.ctrl_dt / a.gait_period) % (2 * np.pi)
            obs = O.build_actor_obs(q_read - zm.home_q, dq, last_a, cmd, ph, *imu)
            act = np.clip(policy(obs), -10, 10)
            last_a = act
            target = np.clip(zm.home_q + a.action_scale * act, zm.lower, zm.upper)
            if k and k % push_every == 0:                       # deterministic push sequence, alternating sides
                ang = rng.uniform(0, 2 * np.pi)
                d.qvel[0] += a.push * np.cos(ang)
                d.qvel[1] += a.push * np.sin(ang)
            for j in range(nsub):
                frac = min(max((j - lat + 1) / nsub, 0.0), 1.0)
                cap = sv.vmax * m.opt.timestep * max(j - lat + 1, 0)
                d.ctrl[zm.act_ids] = target_prev + np.clip((target - target_prev) * frac, -cap, cap)
                mujoco.mj_step(m, d)
                q_buf.append(d.sensordata[_q(zm)].copy())
            target_prev = d.ctrl[zm.act_ids].copy()
            err = target_prev - d.sensordata[_q(zm)]
            track_sq += float(np.sum(err ** 2))
            track_n += err.size
            g = O.gravity_from_quat(zm.sensor(d.sensordata, "pelvis_quat"))
            tilt = np.degrees(np.arccos(np.clip(-g[2], -1, 1)))
            tilts.append(tilt)
            if k * a.ctrl_dt >= 1.0:
                yaw = float(O.yaw_from_quat(zm.sensor(d.sensordata, "pelvis_quat")))
                v = zm.sensor(d.sensordata, "pelvis_linvel")
                vh = np.array([np.cos(yaw) * v[0] + np.sin(yaw) * v[1], -np.sin(yaw) * v[0] + np.cos(yaw) * v[1]])
                vel_err.append(float(np.linalg.norm(vh - np.asarray(a.command)[:2])))
                vx.append(float(vh[0]))
            eq_max = max(eq_max, float(robot.eq_residual(zm, d.qpos).max()))
            gap_max = max(gap_max, float(np.abs(robot.loop_gap(zm, d.sensordata) - zm.loop_gap_home).max()))
            rec = getattr(a, "record", None)
            if rec is not None:  # studio replay: pose every control step (base qpos + the 15 servo angles), push markers
                rec["base"].append(d.qpos[:7].copy())
                rec["q"].append(d.sensordata[_q(zm)].copy())
                rec["tilt"].append(float(tilt))
                if k and k % push_every == 0:
                    rec["pushes"].append(k * a.ctrl_dt)
            if tilt > np.degrees(0.8) or zm.sensor(d.sensordata, "pelvis_pos")[2] < 0.14 or not np.isfinite(d.qpos).all():
                if not fell:
                    falls += 1
                    fall_t.append(k * a.ctrl_dt)
                    fell = True
                    if rec is not None:
                        rec["fell_at"] = k * a.ctrl_dt
                post_fall += 1
                if rec is None or post_fall > 30 or not np.isfinite(d.qpos).all():  # replay keeps 0.6 s of the fall
                    break
    return dict(config=name, dt_ms=m.opt.timestep * 1e3, integrator=c["integrator"].name.replace("mjINT_", ""),
                collisions=c["collisions"], episodes=a.episodes, fall_rate=falls / a.episodes,
                mean_fall_s=float(np.mean(fall_t)) if fall_t else float("nan"),
                track_rms_deg=float(np.degrees(np.sqrt(track_sq / max(track_n, 1)))),
                tilt_mean_deg=float(np.mean(tilts)), tilt_max_deg=float(np.max(tilts)),
                eq_resid_max_deg=float(np.degrees(eq_max)), loop_gap_max_mm=gap_max * 1e3,
                vel_err_mps=float(np.mean(vel_err)) if vel_err else float("nan"),
                mean_vx_mps=float(np.mean(vx)) if vx else float("nan"), wall_s=time.time() - t0)


def _q(zm):
    s0 = zm.sensor_slices[f"q_{zm.joint_names[0]}"].start
    return slice(s0, s0 + zm.nj)


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="home", help="path to policy.onnx, or 'home' (zero action)")
    ap.add_argument("--configs", default="train,sim2sim")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--push", type=float, default=0.15, help="pelvis velocity kick, m/s")
    ap.add_argument("--push-interval", type=float, default=2.5)
    ap.add_argument("--latency", type=float, default=0.010)
    ap.add_argument("--q-delay", type=float, default=0.012)
    ap.add_argument("--ctrl-dt", type=float, default=0.02)
    ap.add_argument("--action-scale", type=float, default=0.3)
    ap.add_argument("--command", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "WZ"),
                    help="velocity command (m/s, m/s, rad/s) for a walking policy; ignored by stand policies")
    ap.add_argument("--gait-period", type=float, default=0.6, help="gait clock period in s (EnvCfg.gait_period_s)")
    ap.add_argument("--imu-body", default="chest")
    ap.add_argument("--no-noise", dest="noise", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imu", choices=["auto", "on", "off"], default="auto",
                    help="actor observation layout; auto = policy_meta.json next to the onnx (home policy: off)")
    a = ap.parse_args(argv)
    meta = os.path.join(os.path.dirname(os.path.abspath(a.policy)), "policy_meta.json")
    if a.imu == "auto":
        a.imu = a.policy != "home" and os.path.exists(meta) and json.load(open(meta)).get("imu", False)
    else:
        a.imu = a.imu == "on"
    a.walk = a.policy != "home" and os.path.exists(meta) and json.load(open(meta)).get("task") == "joystick"
    return a


def main():
    a = parse()
    sv, nz = ServoCfg.load(), NoiseCfg()
    pol = Policy(a.policy)
    rows = [run_config(c, pol, a, sv, nz) for c in a.configs.split(",")]
    print(f"policy: {a.policy}   {a.episodes} episodes x {a.seconds:.0f} s, push {a.push} m/s every {a.push_interval} s, "
          f"noise {'on' if a.noise else 'off'}, IMU in obs: {'yes' if a.imu else 'no'}, "
          f"command {a.command if a.walk else 'none (stand policy)'}")
    hdr = ["config", "dt_ms", "integrator", "collisions", "fall_rate", "mean_fall_s", "track_rms_deg",
           "tilt_mean_deg", "tilt_max_deg", "eq_resid_max_deg", "loop_gap_max_mm", "vel_err_mps", "mean_vx_mps",
           "wall_s"]
    print("  ".join(f"{h:>14s}" for h in hdr))
    for r in rows:
        print("  ".join(f"{r[h]:>14.3f}" if isinstance(r[h], float) else f"{str(r[h]):>14s}" for h in hdr))


if __name__ == "__main__":
    main()
