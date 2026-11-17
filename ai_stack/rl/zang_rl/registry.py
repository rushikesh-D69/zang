"""Task registry, Playground style: env id -> (env config factory, rsl_rl runner config)."""
from __future__ import annotations

import dataclasses

from mujoco_playground._src.rl.rsl_rl import rsl_config

from .config import EnvCfg


@dataclasses.dataclass
class ZangRunnerCfg(rsl_config.RslRlOnPolicyRunnerCfg):
    """PPO settings sized for a 4-core CPU: few worlds, longer rollouts per update."""
    obs_groups: dict = dataclasses.field(default_factory=lambda: {"actor": ["policy"], "critic": ["critic"]})
    nworld: int = 64                 # ~1.2 GB RSS total on CPU (5.5 MB per world model copy)
    max_iterations: int = 3000
    num_steps_per_env: int = 48
    save_interval: int = 100
    experiment_name: str = "zang"

    def __post_init__(self):
        # actor small enough for the ESP32-P4 (50|56-128-128-15, ~25 k MACs, see export/P4_POLICY.md)
        self.actor.hidden_dims = (128, 128)
        self.actor.activation = "elu"
        self.actor.obs_normalization = True
        self.actor.distribution_cfg = rsl_config.RslRlDistributionCfg(init_std=0.8)
        self.critic.hidden_dims = (256, 256, 128)
        self.critic.activation = "elu"
        self.critic.obs_normalization = True
        # fixed LR: with the adaptive (KL) schedule the LR sat at its 1e-5 floor for whole runs (measured,
        # README "Training notes"); a fixed 3e-4 learned faster in the same budget.
        self.algorithm.learning_rate = 3e-4
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.005
        self.algorithm.num_mini_batches = 4
        self.algorithm.num_learning_epochs = 5
        self.algorithm.gamma = 0.99
        self.algorithm.lam = 0.95
        self.algorithm.desired_kl = 0.01


REGISTRY = {
    "ZangStand": (lambda **kw: EnvCfg(task="stand", **kw), ZangRunnerCfg),
    "ZangStandIMU": (lambda **kw: EnvCfg(task="stand", imu=True, **kw),
                     lambda: ZangRunnerCfg(experiment_name="zang_imu")),
    "ZangJoystick": (lambda **kw: EnvCfg(task="joystick", episode_s=20.0, **kw),
                     lambda: ZangRunnerCfg(max_iterations=10000, experiment_name="zang_joystick")),
}


def make(env_id: str, **env_kwargs):
    if env_id not in REGISTRY:
        raise KeyError(f"unknown env {env_id}; available: {list(REGISTRY)}")
    env_fn, runner_fn = REGISTRY[env_id]
    return env_fn(**env_kwargs), runner_fn()
