"""Backend parity: the GPU (MJX/JAX) env must be the same task as the CPU envs.

  .venv/bin/python tests/test_backend_parity.py          # all checks, prints numbers, exit 1 on failure
  .venv/bin/python -m pytest tests/test_backend_parity.py # same, if pytest is installed

1. layouts:  observation / critic / action sizes and block order, joint order, home pose and action convention
             are identical in ZangEnv (rsl_rl, CPU), ZangGymEnv (Gymnasium, CPU) and ZangMjxEnv (JAX), with and
             without the IMU; and they match zang_rl/obs.py, which export_policy.py writes into policy_meta.json.
2. core math: every shared zang_rl/core.py function gives the same numbers with numpy and jax.numpy on the same
             random inputs (domain randomisation, initial pose, sensor setup, servo profile, observation,
             rewards, commands, pushes).
3. rollout:  same start (no randomisation, noise or pushes), same 3 s action sequence: MJX vs CPU MuJoCo joint,
             pelvis and observation trajectories stay close; the parallelogram loops hold in the MJX rollout.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from zang_rl import core, robot  # noqa: E402
from zang_rl import obs as O  # noqa: E402
from zang_rl.config import EnvCfg  # noqa: E402

RESULTS = {}


def test_layouts():
    import gymnasium as gym
    import torch

    import zang_rl  # noqa: F401
    from zang_rl.env import ZangEnv
    from zang_rl.mjx_env import ZangMjxEnv
    table = robot.load_joint_table()
    names = [j["name"] for j in table["joints"]]
    for imu in (False, True):
        n_obs = O.obs_dim(imu)
        cpu = ZangEnv(EnvCfg(nworld=2, imu=imu))
        o = cpu.get_observations()
        g = gym.make("ZangStand-v0", imu=imu).unwrapped
        mx = ZangMjxEnv(task="stand", imu=imu)
        assert o["policy"].shape[1] == n_obs == g.observation_space.shape[0] == mx.observation_size["state"][0]
        assert o["critic"].shape[1] == core.critic_dim() == mx.observation_size["privileged_state"][0]
        assert cpu.num_actions == g.action_space.shape[0] == mx.action_size == 15
        assert float(g.action_space.low.min()) == -1.0 and float(g.action_space.high.max()) == 1.0
        for e in (cpu, g, mx):
            assert e.zm.joint_names == names, "joint order differs from robot.json"
            assert np.allclose(e.zm.home_q, cpu.zm.home_q) and np.array_equal(e.zm.act_ids, cpu.zm.act_ids)
            assert e.zm.sensor_slices == cpu.zm.sensor_slices, "sensordata layout differs"
        assert g.cfg.gym_action_gain == mx.cfg.gym_action_gain
        lay = O.layout(imu)
        assert list(lay) == (["gyro", "gravity"] if imu else []) + ["q", "dq", "last_action", "command", "phase"]
        assert lay["phase"].stop == n_obs
        del cpu, g, mx
        torch.set_num_threads(1)
    RESULTS["layouts"] = "obs 50 (IMU off) / 56 (IMU on), critic 68, action 15, same joint order, sensors, home, gain"


def test_core_math():
    rng = np.random.default_rng(0)
    cfg = EnvCfg(task="joystick")
    zm = robot.build(feet="box")
    base = core.capture_base(zm.model)
    nj, nsub, B = zm.nj, 10, 4
    worst = {}

    def cmp(name, a, b, atol):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        worst[name] = max(worst.get(name, 0.0), err)
        assert a.shape == b.shape and err <= atol, f"{name}: max |numpy - jax| = {err} > {atol}"

    for trial in range(5):
        u = rng.uniform(size=core.dr_num_uniforms(base, nj))
        fn, cn, ln, sn = core.dr_from_uniforms(np, base, zm, cfg, u)
        fj, cj, lj, sj = core.dr_from_uniforms(jnp, base, zm, cfg, jnp.asarray(u, jnp.float32))
        for k in core.DR_FIELDS:
            cmp(f"dr.{k}", fn[k], fj[k], 1e-5 * max(1.0, float(np.abs(fn[k]).max())))
        cmp("dr.calib", cn, cj, 1e-6), cmp("dr.summary", sn, sj, 1e-5)
        assert int(ln) == int(lj), "latency substeps differ"
        u = rng.uniform(size=core.init_num_uniforms(zm))
        cmp("init_qpos", core.init_qpos_from_uniforms(np, zm, cfg, u),
            core.init_qpos_from_uniforms(jnp, zm, cfg, jnp.asarray(u, jnp.float32)), 1e-5)
        u = rng.uniform(size=(B, 7))
        sn_, sj_ = (core.sensor_episode_from_uniforms(x, cfg, x.asarray(u), nsub, 0.002) for x in (np, jnp))
        cmp("sensor.bias", sn_[0], sj_[0], 1e-6), cmp("sensor.R", sn_[1], sj_[1], 1e-6)
        assert np.array_equal(np.asarray(sn_[2]), np.asarray(sj_[2]))
        tp, a = rng.normal(0, 0.3, (B, nj)), rng.normal(0, 1, (B, nj))
        lat = rng.integers(0, 12, B)
        tn = core.action_to_target(np, zm, cfg, a)
        cmp("servo_profile", core.servo_profile(np, tp, tn, lat, nsub, 0.002, cfg.servo),
            core.servo_profile(jnp, jnp.asarray(tp), jnp.asarray(tn), jnp.asarray(lat), nsub, 0.002, cfg.servo), 1e-5)
        sens = rng.normal(0, 0.3, (B, zm.model.nsensordata))
        for q in ("imu_quat", "pelvis_quat"):
            qq = rng.normal(size=(B, 4))
            sens[:, zm.sensor_slices[q]] = qq / np.linalg.norm(qq, axis=1, keepdims=True)
        q_hist = rng.normal(0, 0.3, (B, 2 * nsub, nj))
        args = [sens, q_hist, rng.integers(0, 2 * nsub, B), rng.normal(0, .03, (B, nj)), rng.normal(0, .3, (B, nj)),
                sn_[1], sn_[0], a, rng.normal(0, .1, (B, 3)), rng.uniform(0, 6, B), rng.uniform(0.5, 1.5, (B, 4))]
        noise = {"q": rng.normal(size=(B, nj)), "gyro": rng.normal(size=(B, 3)), "grav": rng.normal(size=(B, 3))}
        on = core.observe(np, zm, cfg, *args, noise)
        oj = core.observe(jnp, zm, cfg, *[jnp.asarray(x) for x in args], {k: jnp.asarray(v) for k, v in noise.items()})
        cmp("obs.actor", on[0], oj[0], cfg.noise.q_quant / cfg.ctrl_dt + 1e-4)   # one quantum on dq at most
        cmp("obs.critic", on[1], oj[1], 1e-4)
        rargs = [sens, rng.uniform(size=B) > 0.1, a, rng.normal(size=(B, nj)), rng.normal(0, .1, (B, 3)),
                 rng.uniform(0, 6, B)]
        rn = core.rewards(np, zm, cfg, *rargs)
        rj = core.rewards(jnp, zm, cfg, *[jnp.asarray(x) for x in rargs])
        cmp("rewards.total", rn[0], rj[0], 1e-4)
        for k in rn[1]:
            cmp(f"rewards.{k}", rn[1][k], rj[1][k], 1e-4 * max(1.0, float(np.abs(rn[1][k]).max())))
        assert np.array_equal(rn[2], np.asarray(rj[2]))
        u = rng.uniform(size=(B, 5))
        for x, y in zip(core.commands_from_uniforms(np, cfg, u), core.commands_from_uniforms(jnp, cfg, jnp.asarray(u))):
            cmp("commands", x, y, 1e-6)
        u = rng.uniform(size=(B, 6))
        for x, y in zip(core.push_from_uniforms(np, cfg, u), core.push_from_uniforms(jnp, cfg, jnp.asarray(u))):
            cmp("push", x, y, 1e-5)
    RESULTS["core_math"] = {k: f"{v:.1e}" for k, v in sorted(worst.items(), key=lambda kv: -kv[1])[:6]}


def test_rollout_vs_cpu(steps=150):
    """Same start, same actions: MJX (float32, JAX) vs CPU MuJoCo (float64) through each backend's own env."""
    import torch

    from zang_rl.env import ZangEnv
    from zang_rl.mjx_env import ZangMjxEnv
    cfg = EnvCfg(task="stand", nworld=1, nthread=1, init_noise_q=0.0, init_yaw=0.0)
    cfg.dr.enabled = cfg.noise.enabled = cfg.push.enabled = False
    cpu = ZangEnv(cfg)
    mx = ZangMjxEnv(cfg=cfg)
    reset, step = jax.jit(mx.reset), jax.jit(mx.step)
    s = reset(jax.random.PRNGKey(0))
    nq = cpu.zm.model.nq
    q0_cpu = cpu.state[0, cpu.qpos_off:cpu.qpos_off + nq]
    assert np.allclose(q0_cpu, np.asarray(s.pipeline_state.qpos), atol=1e-5), "different initial state"
    t = np.arange(steps) * cfg.ctrl_dt
    phase = np.linspace(0, np.pi, 15)
    acts = 0.35 * np.sin(2 * np.pi * 0.8 * t[:, None] + phase[None]) * (np.arange(15) >= 9)   # arms + light legs
    acts[:, 1:9] = 0.08 * np.sin(2 * np.pi * 0.5 * t[:, None] + phase[None, 1:9])
    acts[:, 0] = 0.3 * np.sin(2 * np.pi * 0.4 * t)
    err_q, err_z, err_xy, err_obs, eq_max, gap_max, tilt_cpu, tilt_mx = [], [], [], [], 0.0, 0.0, [], []
    t0 = time.time()
    for k in range(steps):
        o_cpu = cpu.step(torch.as_tensor(acts[k:k + 1] * cfg.gym_action_gain, dtype=torch.float32))[0]
        s = step(s, jnp.asarray(acts[k], jnp.float32))
        qc = cpu.state[0, cpu.qpos_off:cpu.qpos_off + nq]
        qm = np.asarray(s.pipeline_state.qpos)
        err_q.append(np.abs(qc[cpu.zm.qpos_adr] - qm[cpu.zm.qpos_adr]).max())
        err_z.append(abs(qc[2] - qm[2]))
        err_xy.append(np.linalg.norm(qc[:2] - qm[:2]))
        err_obs.append(np.abs(o_cpu["policy"][0].numpy() - np.asarray(s.obs["state"])).max())
        eq_max = max(eq_max, float(robot.eq_residual(mx.zm, qm).max()))
        gap_max = max(gap_max, float(np.abs(robot.loop_gap(mx.zm, np.asarray(s.pipeline_state.sensordata))
                                             - mx.zm.loop_gap_home).max()))
        for qq, out in ((qc, tilt_cpu), (qm, tilt_mx)):
            g = O.gravity_from_quat(qq[3:7])
            out.append(np.degrees(np.arccos(np.clip(-g[2], -1, 1))))
        assert not bool(s.done), "MJX episode ended during the comparison"
    RESULTS["rollout"] = dict(
        steps=steps, seconds=steps * cfg.ctrl_dt, wall_s=round(time.time() - t0, 1),
        joint_max_err_deg=round(float(np.degrees(np.max(err_q))), 3),
        joint_rms_err_deg=round(float(np.degrees(np.sqrt(np.mean(np.square(err_q))))), 3),
        pelvis_z_max_err_mm=round(float(np.max(err_z)) * 1e3, 3),
        pelvis_xy_drift_diff_mm=round(float(np.max(err_xy)) * 1e3, 3),
        obs_max_abs_diff=round(float(np.max(err_obs)), 4),
        tilt_max_deg_cpu=round(float(np.max(tilt_cpu)), 3), tilt_max_deg_mjx=round(float(np.max(tilt_mx)), 3),
        mjx_loop_eq_residual_max_deg=round(float(np.degrees(eq_max)), 3),
        mjx_loop_gap_change_max_mm=round(gap_max * 1e3, 3),
    )
    r = RESULTS["rollout"]
    assert r["joint_max_err_deg"] < 1.0 and r["pelvis_z_max_err_mm"] < 2.0 and r["pelvis_xy_drift_diff_mm"] < 10.0, r
    assert r["mjx_loop_eq_residual_max_deg"] < 1.0 and r["mjx_loop_gap_change_max_mm"] < 1.0, r


if __name__ == "__main__":
    failed = False
    for fn in (test_layouts, test_core_math, test_rollout_vs_cpu):
        try:
            fn()
            print(f"PASS {fn.__name__}: {RESULTS.get(fn.__name__.replace('test_', ''), '')}")
        except AssertionError as e:
            failed = True
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
