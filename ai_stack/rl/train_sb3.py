"""Train ZangStand-v0 / ZangWalk-v0 (Gymnasium) with Stable-Baselines3 PPO.

  .venv/bin/python train_sb3.py --env ZangStand-v0 --steps 2000000 --n-envs 4
  .venv/bin/python train_sb3.py --env ZangStand-v0 --steps 20000 --n-envs 4      # smoke test

Same task, randomisation and observation as the rsl_rl path (both use zang_rl/core.py). One MuJoCo world per
SubprocVecEnv worker, so throughput scales with CPU cores, not GPU. Writes logs/sb3/<env>/<timestamp>/:
progress.csv (learning curves), checkpoints/, final_model.zip, vecnormalize.pkl, env_cfg.json and, at the end,
policy.onnx + p4/policy_weights.h via export_policy.py.
"""
import argparse
import dataclasses
import datetime
import json
import os
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import zang_rl  # noqa: F401  (registers the Gymnasium ids)
from zang_rl.config import EnvCfg

HERE = os.path.dirname(os.path.abspath(__file__))


class SaveVecNormalize(CheckpointCallback):
    """Checkpoint the model and the observation normaliser together (both are needed to run the policy)."""

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.get_vec_normalize_env().save(os.path.join(self.save_path, f"vecnormalize_{self.num_timesteps}.pkl"))
        return super()._on_step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="ZangStand-v0", choices=["ZangStand-v0", "ZangWalk-v0"])
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--n-steps", type=int, default=512, help="rollout length per env per update")
    ap.add_argument("--imu", action="store_true", help="IMU in the observation (future board revision)")
    ap.add_argument("--device", default="cpu", help="cpu is usually faster for small MLP policies")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=200_000, help="env steps between checkpoints")
    ap.add_argument("--logdir", default=os.path.join(HERE, "logs", "sb3"))
    ap.add_argument("--dummy", action="store_true", help="DummyVecEnv (single process, for debugging)")
    ap.add_argument("--start-method", default="fork", choices=["fork", "forkserver", "spawn"])
    a = ap.parse_args()

    torch.set_num_threads(1)
    run_dir = os.path.join(a.logdir, a.env, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    task = "stand" if a.env.startswith("ZangStand") else "joystick"
    with open(os.path.join(run_dir, "env_cfg.json"), "w") as f:
        json.dump(dataclasses.asdict(EnvCfg(task=task, imu=a.imu, nworld=1)), f, indent=2)

    # fork: workers share the parent's already-imported torch/SB3 pages (copy-on-write). With the default
    # forkserver every worker re-imports torch (~0.8 GB PSS each).
    venv = make_vec_env(a.env, n_envs=a.n_envs, seed=a.seed, env_kwargs={"imu": a.imu},
                        vec_env_cls=DummyVecEnv if a.dummy else SubprocVecEnv,
                        vec_env_kwargs=None if a.dummy else {"start_method": a.start_method})
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=100.0, gamma=0.99)
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[256, 256, 128]), activation_fn=torch.nn.ELU,
                         log_std_init=-1.0)
    model = PPO("MlpPolicy", venv, n_steps=a.n_steps, batch_size=min(2048, a.n_steps * a.n_envs), n_epochs=5,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, max_grad_norm=1.0,
                policy_kwargs=policy_kwargs, seed=a.seed, device=a.device, verbose=0)
    model.set_logger(configure(run_dir, ["csv", "stdout", "tensorboard"]))
    ckpt = SaveVecNormalize(save_freq=max(a.save_every // a.n_envs, 1), save_path=os.path.join(run_dir, "checkpoints"),
                            name_prefix="model")
    t = time.time()
    model.learn(total_timesteps=a.steps, callback=CallbackList([ckpt]), progress_bar=False)
    el = time.time() - t
    model.save(os.path.join(run_dir, "final_model.zip"))
    venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
    venv.close()
    print(f"done: {model.num_timesteps} env steps in {el:.0f} s ({model.num_timesteps / el:.0f} steps/s) -> {run_dir}")

    from export_policy import export
    export(os.path.join(run_dir, "final_model.zip"), run_dir)


if __name__ == "__main__":
    main()
