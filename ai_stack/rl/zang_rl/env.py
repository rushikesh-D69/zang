"""rsl_rl VecEnv for Zang (ZangStand / ZangStandIMU / ZangJoystick): many worlds on the CPU via mujoco.rollout.

The task itself (randomisation, command path, observations, rewards, pushes) lives in core.py and is shared with
the Gymnasium env (gym_env.py). This file only does the batching: each world owns its MjModel copy (per-world
domain randomisation) and `mujoco.rollout` steps all worlds in parallel C threads.

Per policy step (50 Hz, 10 physics substeps at 2 ms):
  action -> joint target -> per-substep targets (latency, DS ramp, speed cap) + servo zero error -> rollout ->
  per-substep sensordata -> delayed / noisy servo-bus read (+ IMU if enabled) -> rewards -> resets, pushes.
"""
from __future__ import annotations

import copy
import os

import mujoco
import numpy as np
import torch
from mujoco import rollout
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from . import core, robot
from .config import EnvCfg


class ZangEnv(VecEnv):
    def __init__(self, cfg: EnvCfg, device: str = "cpu", seed: int = 0):
        self.cfg, self.device = cfg, device
        self.rng = np.random.default_rng(seed)
        self.zm = zm = robot.build(imu_body=cfg.imu_body, collisions=cfg.collisions)
        m0 = zm.model
        self.dt = float(m0.opt.timestep)
        self.nsub = int(round(cfg.ctrl_dt / self.dt))
        assert abs(self.nsub * self.dt - cfg.ctrl_dt) < 1e-9, "ctrl_dt must be a multiple of the timestep"
        B, nj = cfg.nworld, zm.nj
        self.num_envs, self.num_actions = B, nj
        self.max_episode_length = int(round(cfg.episode_s / cfg.ctrl_dt))
        self.episode_length_buf = torch.zeros(B, dtype=torch.long)

        self.base = core.capture_base(m0)
        self.models = [copy.copy(m0) for _ in range(B)]
        nthread = cfg.nthread or os.cpu_count() or 1
        self.datas = [mujoco.MjData(m0) for _ in range(nthread)]
        self.scratch = mujoco.MjData(m0)
        self.roller = rollout.Rollout(nthread=nthread)
        self.state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.nstate = mujoco.mj_stateSize(m0, self.state_spec)
        self.qpos_off, self.qvel_off = 1, 1 + m0.nq        # FULLPHYSICS = [time, qpos, qvel, act, ...]
        self.state = np.zeros((B, self.nstate))
        self.out_state = np.zeros((B, self.nsub, self.nstate))
        self.out_sens = np.zeros((B, self.nsub, m0.nsensordata))
        self.ctrl = np.zeros((B, self.nsub, m0.nu))
        self.q_sl = core.q_sensor_slice(zm, "q")

        self.sens = np.zeros((B, m0.nsensordata))
        self.q_hist = np.zeros((B, 2 * self.nsub, nj))
        self.target_prev = np.tile(zm.home_q, (B, 1))
        self.last_action = np.zeros((B, nj))
        self.prev_action = np.zeros((B, nj))
        self.q_read_prev = np.zeros((B, nj))
        self.command = np.zeros((B, 3))
        self.cmd_timer = np.zeros(B)
        self.phase = np.zeros(B)
        self.push_timer = np.zeros(B)
        self.gyro_bias = np.zeros((B, 3))
        self.imu_R = np.tile(np.eye(3), (B, 1, 1))
        self.calib = np.zeros((B, nj))
        self.lat_sub = np.zeros(B, dtype=int)
        self.qdelay_sub = np.zeros(B, dtype=int)
        self.dr_summary = np.ones((B, 4))
        self.reward_names = list(cfg.reward_scales)
        self._ep_sums = {k: np.zeros(B) for k in self.reward_names}
        self._log = {}
        self.reset_idx(np.arange(B))
        self._obs = self._compute_obs()

    # ======================================================================================================== API
    def get_observations(self) -> TensorDict:
        return self._obs

    def reset(self):
        self.reset_idx(np.arange(self.num_envs))
        self._obs = self._compute_obs()
        return self._obs, {}

    def step(self, actions: torch.Tensor):
        cfg, zm = self.cfg, self.zm
        a = np.clip(actions.detach().cpu().numpy().astype(np.float64), -10.0, 10.0)
        self.prev_action, self.last_action = self.last_action, a
        prof = core.servo_profile(np, self.target_prev, core.action_to_target(np, zm, cfg, a), self.lat_sub,
                                  self.nsub, self.dt, cfg.servo)
        self.target_prev = prof[:, -1].copy()
        self.ctrl[:, :, zm.act_ids] = prof + self.calib[:, None, :]

        self.roller.rollout(self.models, self.datas, self.state, self.ctrl, control_spec=mujoco.mjtState.mjSTATE_CTRL,
                            skip_checks=True, nstep=self.nsub, state=self.out_state, sensordata=self.out_sens)
        self.state[:] = self.out_state[:, -1]
        self.sens[:] = self.out_sens[:, -1]
        self.q_hist = np.concatenate([self.q_hist[:, self.nsub:], self.out_sens[:, :, self.q_sl]], 1)

        self.episode_length_buf += 1
        self.phase = core.advance_phase(np, cfg, self.phase)
        if cfg.task == "joystick":
            self.cmd_timer -= cfg.ctrl_dt
            ids = np.flatnonzero(self.cmd_timer <= 0)
            if len(ids):
                self.command[ids], self.cmd_timer[ids] = core.commands_from_uniforms(
                    np, cfg, self.rng.uniform(size=(len(ids), 5)))

        rew, terms, terminated = core.rewards(np, zm, cfg, self.sens, np.isfinite(self.state).all(-1), self.last_action,
                                              self.prev_action, self.command, self.phase)
        timeouts = self.episode_length_buf.numpy() >= self.max_episode_length
        dones = terminated | timeouts
        for k, v in terms.items():
            self._ep_sums[k] += v

        self._log = {}
        done_ids = np.flatnonzero(dones)
        if len(done_ids):
            for k in self.reward_names:
                self._log[f"/episode_reward/{k}"] = float(self._ep_sums[k][done_ids].mean())
                self._ep_sums[k][done_ids] = 0.0
            self._log["/termination/fall_rate"] = float(terminated[done_ids].mean())
            self._log["/termination/mean_len_s"] = float(self.episode_length_buf.numpy()[done_ids].mean() * cfg.ctrl_dt)
            self.reset_idx(done_ids)
        if cfg.push.enabled:
            self.push_timer -= cfg.ctrl_dt
            ids = np.flatnonzero(self.push_timer <= 0)
            if len(ids):
                dv, self.push_timer[ids] = core.push_from_uniforms(np, cfg, self.rng.uniform(size=(len(ids), 6)))
                self.state[ids, self.qvel_off:self.qvel_off + 6] += dv
        self._obs = self._compute_obs()
        extras = {"time_outs": torch.as_tensor(timeouts & ~terminated), "log": self._log}
        return (self._obs, torch.as_tensor(rew, dtype=torch.float32),
                torch.as_tensor(dones, dtype=torch.long), extras)

    def _compute_obs(self) -> TensorDict:
        actor, critic, self.q_read_prev = core.observe(
            np, self.zm, self.cfg, self.sens, self.q_hist, self.qdelay_sub, self.calib, self.q_read_prev,
            self.imu_R, self.gyro_bias, self.last_action, self.command, self.phase, self.dr_summary,
            core.normals_numpy(self.rng, self.num_envs, self.zm.nj))
        return TensorDict({"policy": torch.from_numpy(actor), "critic": torch.from_numpy(critic)},
                          batch_size=[self.num_envs])

    # ======================================================================================================= reset
    def reset_idx(self, ids: np.ndarray):
        cfg = self.cfg
        for i in ids:
            i = int(i)
            self.calib[i], self.lat_sub[i], self.dr_summary[i] = core.reset_world_numpy(
                self.models[i], self.scratch, self.base, self.zm, cfg, self.rng)
            st = np.empty(self.nstate)
            mujoco.mj_getState(self.models[i], self.scratch, st, self.state_spec)
            self.state[i] = st
            self.sens[i] = self.scratch.sensordata
        n = len(ids)
        self.episode_length_buf[torch.as_tensor(ids)] = 0
        self.last_action[ids] = 0
        self.prev_action[ids] = 0
        self.phase[ids] = self.rng.uniform(0, 2 * np.pi, n)
        self.push_timer[ids] = self.rng.uniform(*cfg.push.interval_s, n)
        self.gyro_bias[ids], self.imu_R[ids], self.qdelay_sub[ids] = core.sensor_episode_from_uniforms(
            np, cfg, self.rng.uniform(size=(n, 7)), self.nsub, self.dt)
        q_true = self.sens[ids][:, self.q_sl]
        self.q_read_prev[ids] = q_true - self.calib[ids]
        self.q_hist[ids] = q_true[:, None, :]
        self.target_prev[ids] = q_true
        if cfg.task == "joystick":
            self.command[ids], self.cmd_timer[ids] = core.commands_from_uniforms(np, cfg, self.rng.uniform(size=(n, 5)))
