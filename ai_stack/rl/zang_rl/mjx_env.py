"""MJX (JAX) Zang env for GPU training with brax PPO: thousands of worlds on one GPU (e.g. a Colab T4).

Same task as the CPU envs: every piece of task math (domain randomisation, initial state, servo command path,
IMU-free observation, rewards, terminations, commands, pushes) is the shared zang_rl/core.py called with
xp = jax.numpy. Only the stepping differs: `mjx.step` inside `jax.lax.scan` over the physics substeps.

Model: zang_rl.robot.build(collisions="feet", feet="box"): box feet against the floor plane (exact, cheap,
supported by MJX), no self collision, the parallelogram loops as joint-coupling equalities (unchanged).

Action: brax PPO with a tanh-normal policy emits a in [-1, 1]; the env applies cfg.gym_action_gain (same
convention as the Gymnasium env), so joint target = home + action_scale * gain * a.

Auto-reset happens inside step() (not brax's AutoResetWrapper, which would restore only pipeline_state and obs
and keep the stale per-episode randomisation, calibration and servo buffers in `info`). Use `wrap_for_brax`
as brax's `wrap_env_fn`: it only vmaps. Because brax's EpisodeWrapper is not used either, this env keeps the
bookkeeping brax PPO / Evaluator read from `state.info` itself: `steps`, `truncation`, `episode_done`,
`episode_metrics` (same semantics as brax.envs.wrappers.training.EpisodeWrapper: at the step that ends an episode
`steps` still holds its length and `episode_done` is 1; the state returned by that step is already the next
episode's initial state).

Checked against the installed brax 0.14.2 / mujoco-mjx 4.0.1 source (ppo.train, acting.Evaluator, EpisodeWrapper).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from brax.envs.wrappers.training import VmapWrapper
from mujoco import mjx

from . import core, robot
from . import obs as O
from .config import EnvCfg


class ZangMjxEnv(Env):
    def __init__(self, cfg: EnvCfg | None = None, task: str = "stand", **cfg_overrides):
        self.cfg = cfg or EnvCfg(task=task, nworld=1, **cfg_overrides)
        if self.cfg.collisions != "feet":
            raise ValueError("MJX env supports collisions='feet' (box feet vs floor) only")
        self.zm = zm = robot.build(imu_body=self.cfg.imu_body, collisions="feet", feet="box")
        m = zm.model
        self.base = core.capture_base(m)
        self.mx = mjx.put_model(m, impl="jax")        # pure-JAX MJX (also the default on a CUDA GPU)
        self.d0 = mjx.make_data(m, impl="jax")
        self.dt = float(m.opt.timestep)
        self.nsub = int(round(self.cfg.ctrl_dt / self.dt))
        self.episode_length = int(round(self.cfg.episode_s / self.cfg.ctrl_dt))
        self.n_dr = core.dr_num_uniforms(self.base, zm.nj)
        self.q_sl = core.q_sensor_slice(zm, "q")
        self.act_ids = jnp.asarray(zm.act_ids)
        self.foot_sites = jnp.asarray(zm.foot_site_ids)
        self.reward_names = list(self.cfg.reward_scales)

    # ---- brax Env interface ---------------------------------------------------------------------------------
    @property
    def observation_size(self):
        return {"state": (O.obs_dim(self.cfg.imu),), "privileged_state": (core.critic_dim(),)}

    @property
    def action_size(self) -> int:
        return self.zm.nj

    @property
    def backend(self) -> str:
        return "mjx"

    def reset(self, rng: jax.Array) -> State:
        data, info, obs = self._reset_world(rng)
        zero = jnp.zeros(())
        metrics = {f"reward/{k}": zero for k in self.reward_names} | {"fall": zero}
        info["steps"] = jnp.zeros((), jnp.int32)
        info["truncation"] = zero
        info["episode_done"] = zero
        info["episode_metrics"] = {"sum_reward": zero, "length": zero} | {k: zero for k in metrics}
        return State(pipeline_state=data, obs=obs, reward=zero, done=zero, metrics=metrics, info=info)

    def step(self, state: State, action: jax.Array) -> State:
        cfg, zm, info = self.cfg, self.zm, dict(state.info)
        rng, k_obs, k_cmd, k_push, k_reset = jax.random.split(info["rng"], 5)
        a = (jnp.clip(action, -1.0, 1.0) * cfg.gym_action_gain)[None]
        prev_action, last_action = info["last_action"], a
        target = core.action_to_target(jnp, zm, cfg, a)
        prof = core.servo_profile(jnp, info["target_prev"][None], target, info["lat_sub"], self.nsub, self.dt,
                                  cfg.servo)[0]
        model = self.mx.replace(**info["dr"])

        def substep(d, tgt):
            d = d.replace(ctrl=d.ctrl.at[self.act_ids].set(tgt + info["calib"]))
            d = mjx.step(model, d)
            return d, d.sensordata
        data, sens_sub = jax.lax.scan(substep, state.pipeline_state, prof)
        sens = sens_sub[-1][None]                      # last-substep sensordata, as the CPU env uses
        q_hist = jnp.concatenate([info["q_hist"][self.nsub:], sens_sub[:, self.q_sl]], 0)

        phase = core.advance_phase(jnp, cfg, info["phase"])
        command, cmd_timer = info["command"], info["cmd_timer"]
        if cfg.task == "joystick":
            cmd_timer = cmd_timer - cfg.ctrl_dt
            new_cmd, new_timer = core.commands_from_uniforms(jnp, cfg, jax.random.uniform(k_cmd, (1, 5)))
            resample = cmd_timer <= 0
            command = jnp.where(resample, new_cmd[0], command)
            cmd_timer = jnp.where(resample, new_timer[0], cmd_timer)

        finite = jnp.isfinite(data.qpos).all() & jnp.isfinite(data.qvel).all()
        rew, terms, terminated = core.rewards(jnp, zm, cfg, sens, finite[None], last_action, prev_action,
                                              command[None], phase[None])
        steps = jnp.where(info["episode_done"] > 0, 0, info["steps"]) + 1      # restart after an episode ended
        truncation = (steps >= self.episode_length) & ~terminated[0]
        done = terminated[0] | truncation

        push_timer = info["push_timer"]
        if cfg.push.enabled:
            push_timer = push_timer - cfg.ctrl_dt
            dv, new_timer = core.push_from_uniforms(jnp, cfg, jax.random.uniform(k_push, (1, 6)))
            kick = push_timer <= 0
            data = data.replace(qvel=data.qvel.at[:6].add(jnp.where(kick, dv[0], 0.0)))
            push_timer = jnp.where(kick, new_timer[0], push_timer)

        nxt = dict(info, rng=rng, last_action=last_action, prev_action=prev_action,
                   target_prev=prof[-1], q_hist=q_hist, phase=phase, command=command, cmd_timer=cmd_timer,
                   push_timer=push_timer)
        obs, nxt["q_read_prev"] = self._observe(jnp.asarray(sens), nxt, k_obs)

        # ---- brax bookkeeping (kept outside the auto-reset below) -----------------------------------------
        f32 = jnp.float32
        metrics = {f"reward/{k}": v[0] for k, v in terms.items()} | {"fall": terminated[0].astype(f32)}
        inc = {"sum_reward": rew[0], "length": 1.0} | metrics
        book = dict(steps=steps, truncation=truncation.astype(f32), episode_done=done.astype(f32),
                    episode_metrics={k: v * (1.0 - info["episode_done"]) + inc[k]
                                     for k, v in info["episode_metrics"].items()})
        for k in book:
            nxt.pop(k, None)

        # ---- auto-reset inside the env (fresh randomisation for the new episode) --------------------------
        r_data, r_info, r_obs = self._reset_world(k_reset)
        pick = lambda new, old: jax.tree.map(lambda x, y: jnp.where(done, x, y), new, old)
        data = pick(r_data, data)
        obs = pick(r_obs, obs)
        nxt = pick(r_info, nxt) | book
        return state.replace(pipeline_state=data, obs=obs, reward=rew[0], done=done.astype(f32),
                             metrics=metrics, info=nxt)

    # ---- internals ------------------------------------------------------------------------------------------
    def _reset_world(self, key):
        cfg, zm = self.cfg, self.zm
        k_dr, k_q, k_sens, k_cmd, k_ph, k_push, k_obs, k_next = jax.random.split(key, 8)
        fields, calib, lat_sub, summary = core.dr_from_uniforms(jnp, self.base, zm, cfg,
                                                                jax.random.uniform(k_dr, (self.n_dr,)))
        model = self.mx.replace(**fields)
        qpos = core.init_qpos_from_uniforms(jnp, zm, cfg, jax.random.uniform(k_q, (core.init_num_uniforms(zm),)))
        data = mjx.kinematics(model, self.d0.replace(qpos=qpos))
        qpos = qpos.at[2].add(core.SOLE_CLEARANCE - jnp.min(data.site_xpos[self.foot_sites, 2]))
        ctrl = self.d0.ctrl.at[self.act_ids].set(qpos[zm.qpos_adr] + calib)
        data = mjx.forward(model, self.d0.replace(qpos=qpos, ctrl=ctrl))
        bias, R, delay = core.sensor_episode_from_uniforms(jnp, cfg, jax.random.uniform(k_sens, (1, 7)), self.nsub,
                                                           self.dt)
        if cfg.task == "joystick":
            command, timer = core.commands_from_uniforms(jnp, cfg, jax.random.uniform(k_cmd, (1, 5)))
            command, timer = command[0], timer[0]
        else:
            command, timer = jnp.zeros(3), jnp.zeros(())
        q_true = data.sensordata[self.q_sl]
        nj = zm.nj
        info = dict(
            rng=k_next, dr=fields, calib=calib, lat_sub=lat_sub, dr_summary=summary,
            gyro_bias=bias[0], imu_R=R[0], qdelay=delay[0], q_hist=jnp.tile(q_true, (2 * self.nsub, 1)),
            target_prev=q_true, last_action=jnp.zeros((1, nj)), prev_action=jnp.zeros((1, nj)),
            q_read_prev=(q_true - calib)[None], command=command, cmd_timer=timer,
            phase=jax.random.uniform(k_ph, (), minval=0.0, maxval=2 * np.pi),
            push_timer=jax.random.uniform(k_push, (), minval=cfg.push.interval_s[0], maxval=cfg.push.interval_s[1]),
        )
        obs, info["q_read_prev"] = self._observe(data.sensordata[None], info, k_obs)
        return data, info, obs

    def _observe(self, sens, info, key):
        kq, kg, kv = jax.random.split(key, 3)
        noise = {"q": jax.random.normal(kq, (1, self.zm.nj)), "gyro": jax.random.normal(kg, (1, 3)),
                 "grav": jax.random.normal(kv, (1, 3))}
        actor, critic, q_read = core.observe(
            jnp, self.zm, self.cfg, sens, info["q_hist"][None], info["qdelay"][None], info["calib"][None],
            info["q_read_prev"], info["imu_R"][None], info["gyro_bias"][None], info["last_action"],
            info["command"][None], info["phase"][None], info["dr_summary"][None], noise)
        return {"state": actor[0], "privileged_state": critic[0]}, q_read


def wrap_for_brax(env, episode_length=None, action_repeat=1, randomization_fn=None, **_):
    """brax `wrap_env_fn`: vectorise only (episode length, auto-reset and randomisation live in the env)."""
    return VmapWrapper(env)
