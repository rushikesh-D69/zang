"""Train a Zang policy with rsl_rl PPO on the CPU MuJoCo backend.

  .venv/bin/python train.py --env ZangStand                         # full run (hours on a 4-core CPU)
  .venv/bin/python train.py --env ZangStand --iters 5 --nworld 32   # smoke test (~1 min)
  .venv/bin/python train.py --env ZangJoystick --resume logs/ZangStand/<run>/model_3000.pt

Writes logs/<env>/<timestamp>/: TensorBoard events, model_<it>.pt, env_cfg.json, runner_cfg.json, and at the end
policy.onnx (+ policy_meta.json) via export_policy.py.
"""
import argparse
import dataclasses
import datetime
import json
import os
import time

import torch
from rsl_rl.runners import OnPolicyRunner

from zang_rl.env import ZangEnv
from zang_rl.registry import make

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="ZangStand")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--nworld", type=int, default=None)
    ap.add_argument("--steps-per-env", type=int, default=None)
    ap.add_argument("--nthread", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lr-schedule", choices=["adaptive", "fixed"], default=None)
    ap.add_argument("--no-dr", action="store_true", help="disable domain randomisation, noise and pushes (debug only)")
    ap.add_argument("--resume", default=None, help="checkpoint .pt to initialise the actor from")
    ap.add_argument("--logdir", default=os.path.join(HERE, "logs"))
    a = ap.parse_args()

    env_cfg, runner_cfg = make(a.env, nthread=a.nthread)
    if a.nworld:
        runner_cfg.nworld = a.nworld
    if a.iters:
        runner_cfg.max_iterations = a.iters
    if a.steps_per_env:
        runner_cfg.num_steps_per_env = a.steps_per_env
    runner_cfg.seed = a.seed
    if a.lr:
        runner_cfg.algorithm.learning_rate = a.lr
    if a.lr_schedule:
        runner_cfg.algorithm.schedule = a.lr_schedule
    env_cfg.nworld = runner_cfg.nworld
    if a.no_dr:
        env_cfg.dr.enabled = env_cfg.noise.enabled = env_cfg.push.enabled = False

    run_dir = os.path.join(a.logdir, a.env, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "env_cfg.json"), "w") as f:
        json.dump(dataclasses.asdict(env_cfg), f, indent=2)
    with open(os.path.join(run_dir, "runner_cfg.json"), "w") as f:
        json.dump(dataclasses.asdict(runner_cfg), f, indent=2)

    torch.manual_seed(a.seed)
    env = ZangEnv(env_cfg, device="cpu", seed=a.seed)
    runner = OnPolicyRunner(env, dataclasses.asdict(runner_cfg), log_dir=run_dir, device="cpu")
    if a.resume:
        runner.load(a.resume, load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False})
    t = time.time()
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)
    el = time.time() - t
    steps = runner_cfg.max_iterations * runner_cfg.num_steps_per_env * runner_cfg.nworld
    final = os.path.join(run_dir, f"model_{runner_cfg.max_iterations}.pt")
    runner.save(final)
    print(f"done: {steps} env steps in {el:.0f} s ({steps / el:.0f} steps/s), checkpoint {final}")

    from export_policy import export
    export(final, run_dir)


if __name__ == "__main__":
    main()
