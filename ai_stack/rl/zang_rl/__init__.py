"""Zang RL. Importing the package registers the Gymnasium envs (no torch import here).

  import gymnasium as gym, zang_rl
  env = gym.make("ZangStand-v0")                 # IMU-free observation (hardware v0.0.1)
  env = gym.make("ZangStand-v0", imu=True)       # future board with an IMU
  env = gym.make("ZangWalk-v0")                  # velocity-tracking walk
"""
try:
    import gymnasium as _gym

    _EPISODE_STEPS = 1000            # 20 s at 50 Hz (EnvCfg.episode_s / ctrl_dt)
    for _id, _task in (("ZangStand-v0", "stand"), ("ZangWalk-v0", "joystick")):
        if _id not in _gym.registry:
            _gym.register(id=_id, entry_point="zang_rl.gym_env:ZangGymEnv", kwargs={"task": _task},
                          max_episode_steps=_EPISODE_STEPS)
except ImportError:  # gymnasium is optional for the rsl_rl path
    pass
