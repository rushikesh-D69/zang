"""Gymnasium front end: ZangStand-v0 / ZangWalk-v0 as a `gymnasium.envs.mujoco.MujocoEnv` subclass.

Same task as the rsl_rl env (zang_rl/env.py): both call zang_rl/core.py for domain randomisation, the servo
command path (latency, DS ramp, speed cap, zero error), the IMU-free observation, rewards, terminations,
commands and pushes. One world per env; vectorise with SB3 `make_vec_env(..., vec_env_cls=SubprocVecEnv)`.

Why a dedicated MujocoEnv subclass and not `gym.make("Humanoid-v5", xml_file=...)`: Humanoid-v5 drives torque
actuators and observes 348 privileged values (cinert, cvel, cfrc_ext). Zang has position servos with a bus
latency, no IMU, and closed parallelogram loops. The observation must be what the robot can read, and the action
must go through the servo command path. A Reward/Observation wrapper cannot change the per-substep actuation.

API used (checked against the installed gymnasium 1.4.0, gymnasium/envs/mujoco/mujoco_env.py):
  MujocoEnv.__init__(model_path, frame_skip, observation_space, render_mode, width, height, ..., default_camera_config)
  overrides: _initialize_simulation() -> (model, data), _set_action_space(), reset_model(), step()
"""
from __future__ import annotations

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

from . import core, robot
from . import obs as O
from .config import EnvCfg

# Gym actions are normalised to [-1, 1] (SB3 / check_env convention) and multiplied by EnvCfg.gym_action_gain
# before the shared command path: joint target = home + action_scale * gain * a  (default +-0.6 rad around home).


class ZangGymEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"], "render_fps": 50}

    def __init__(self, task: str = "stand", cfg: EnvCfg | None = None, render_mode: str | None = None,
                 width: int = 640, height: int = 480, **cfg_overrides):
        self.cfg = cfg or EnvCfg(task=task, nworld=1, **cfg_overrides)
        # visual meshes only when rendering (75 MB per model otherwise)
        self.zm = robot.build(imu_body=self.cfg.imu_body, collisions=self.cfg.collisions,
                              strip_visuals=render_mode is None)
        timestep = float(self.zm.model.opt.timestep)
        self.nsub = int(round(self.cfg.ctrl_dt / timestep))
        obs_space = spaces.Box(-np.inf, np.inf, (O.obs_dim(self.cfg.imu),), np.float32)
        MujocoEnv.__init__(self, robot.SCENE_XML, self.nsub, observation_space=obs_space, render_mode=render_mode,
                           width=width, height=height,
                           default_camera_config={"trackbodyid": 1, "distance": 0.9, "elevation": -15.0})
        self.base = core.capture_base(self.model)
        self.scratch = mujoco.MjData(self.model)
        self.q_sl = core.q_sensor_slice(self.zm, "q")
        nj = self.zm.nj
        self.q_hist = np.zeros((1, 2 * self.nsub, nj))
        self.target_prev = self.zm.home_q[None].copy()
        self.last_action = np.zeros((1, nj))
        self.prev_action = np.zeros((1, nj))
        self.q_read_prev = np.zeros((1, nj))
        self.command = np.zeros((1, 3))
        self.cmd_timer = np.zeros(1)
        self.phase = np.zeros(1)
        self.push_timer = np.zeros(1)
        self.calib = np.zeros((1, nj))
        self.lat_sub = np.zeros(1, dtype=int)
        self.dr_summary = np.ones((1, 4))
        self.gyro_bias, self.imu_R, self.qdelay_sub = np.zeros((1, 3)), np.eye(3)[None], np.zeros(1, dtype=int)

    # ---- MujocoEnv overrides --------------------------------------------------------------------------------
    def _initialize_simulation(self):
        m = self.zm.model
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, self.width)
        m.vis.global_.offheight = max(m.vis.global_.offheight, self.height)
        return m, mujoco.MjData(m)

    def _set_action_space(self):
        self.action_space = spaces.Box(-1.0, 1.0, (self.zm.nj,), np.float32)
        return self.action_space

    def reset_model(self):
        cfg, zm, rng = self.cfg, self.zm, self.np_random
        self.calib[0], self.lat_sub[0], self.dr_summary[0] = core.reset_world_numpy(
            self.model, self.data, self.base, zm, cfg, rng)
        q_true = self.data.sensordata[self.q_sl].copy()
        self.q_hist[:] = q_true
        self.q_read_prev[0] = q_true - self.calib[0]
        self.target_prev[0] = q_true
        self.last_action[:] = 0
        self.prev_action[:] = 0
        self.phase[0] = rng.uniform(0, 2 * np.pi)
        self.push_timer[0] = rng.uniform(*cfg.push.interval_s)
        self.gyro_bias, self.imu_R, self.qdelay_sub = core.sensor_episode_from_uniforms(
            np, cfg, rng.uniform(size=(1, 7)), self.nsub, self.model.opt.timestep)
        if cfg.task == "joystick":
            self.command, self.cmd_timer = core.commands_from_uniforms(np, cfg, rng.uniform(size=(1, 5)))
        else:
            self.command[:] = 0
        return self._observe()

    def step(self, action):
        cfg, zm, m, d = self.cfg, self.zm, self.model, self.data
        a = np.clip(np.asarray(action, np.float64).reshape(1, -1), -1, 1) * cfg.gym_action_gain
        self.prev_action, self.last_action = self.last_action, a
        prof = core.servo_profile(np, self.target_prev, core.action_to_target(np, zm, cfg, a), self.lat_sub,
                                  self.nsub, m.opt.timestep, cfg.servo)[0]
        q_sub = np.empty((self.nsub, zm.nj))
        for k in range(self.nsub):
            d.ctrl[zm.act_ids] = prof[k] + self.calib[0]
            mujoco.mj_step(m, d)
            q_sub[k] = d.sensordata[self.q_sl]
        self.target_prev = prof[-1][None].copy()
        self.q_hist = np.concatenate([self.q_hist[:, self.nsub:], q_sub[None]], 1)

        self.phase = core.advance_phase(np, cfg, self.phase)
        if cfg.task == "joystick":
            self.cmd_timer -= cfg.ctrl_dt
            if self.cmd_timer[0] <= 0:
                self.command, self.cmd_timer = core.commands_from_uniforms(np, cfg, self.np_random.uniform(size=(1, 5)))
        sens = d.sensordata[None].copy()
        finite = np.array([np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()])
        rew, terms, terminated = core.rewards(np, zm, cfg, sens, finite, self.last_action, self.prev_action,
                                              self.command, self.phase)
        if cfg.push.enabled:
            self.push_timer -= cfg.ctrl_dt
            if self.push_timer[0] <= 0:
                dv, self.push_timer = core.push_from_uniforms(np, cfg, self.np_random.uniform(size=(1, 6)))
                d.qvel[:6] += dv[0]
        obs = self._observe()
        if self.render_mode == "human":
            self.render()
        info = {f"reward/{k}": float(v[0]) for k, v in terms.items()}
        return obs, float(rew[0]), bool(terminated[0]), False, info

    def _observe(self):
        actor, _critic, self.q_read_prev = core.observe(
            np, self.zm, self.cfg, self.data.sensordata[None], self.q_hist, self.qdelay_sub, self.calib,
            self.q_read_prev, self.imu_R, self.gyro_bias, self.last_action, self.command, self.phase, self.dr_summary,
            core.normals_numpy(self.np_random, 1, self.zm.nj))
        return actor[0]
