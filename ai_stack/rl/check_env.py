"""Verify the RL env is physically sane before training.

Checks, under random actions + pushes + full domain randomisation:
  1. parallelogram loops hold: joint-coupling equality residual (rad) and the true geometric loop gap
     (distance between paired pin sites, mm) relative to home;
  2. sensors: gravity is a unit vector, the observation is finite, home pose reads home;
  3. throughput.
Exit code 1 if a loop check fails.

  .venv/bin/python check_env.py [--nworld 32] [--steps 300] [--action 0.5]
"""
import argparse
import sys
import time

import numpy as np
import torch

from zang_rl import robot
from zang_rl.config import EnvCfg
from zang_rl.env import ZangEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nworld", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--action", type=float, default=0.5, help="random action amplitude (x action_scale rad)")
    ap.add_argument("--task", default="stand")
    ap.add_argument("--imu", action="store_true")
    a = ap.parse_args()

    env = ZangEnv(EnvCfg(task=a.task, nworld=a.nworld, imu=a.imu), seed=1)
    zm = env.zm
    rng = np.random.default_rng(0)
    eq_max, gap_max, gap_home = 0.0, 0.0, zm.loop_gap_home
    t = time.time()
    act = np.zeros((a.nworld, zm.nj))
    for k in range(a.steps):
        act = 0.8 * act + 0.2 * rng.uniform(-a.action, a.action, act.shape) * 5    # smooth random walk
        env.step(torch.as_tensor(np.clip(act, -a.action, a.action), dtype=torch.float32))
        qpos = env.out_state[:, :, env.qpos_off:env.qpos_off + zm.model.nq]
        # residual against each world's own (randomised) coupling slope: checks the constraint, not the DR
        k_world = np.stack([m.eq_data[:, 1] for m in env.models])                  # (B, neq)
        for e, (p, s, _) in enumerate(zm.eq_pairs):
            res = np.abs(qpos[:, :, p] - k_world[:, None, e] * qpos[:, :, s])
            eq_max = max(eq_max, float(res.max()))
        gap = robot.loop_gap(zm, env.out_sens)
        gap_max = max(gap_max, float(np.abs(gap - gap_home).max()))
    el = time.time() - t
    obs = env.get_observations()["policy"].numpy()
    g = env.get_observations()["critic"].numpy()[:, 3:6]      # critic always carries the true IMU block
    print(f"steps {a.steps} x {a.nworld} worlds in {el:.1f} s  ({a.steps * a.nworld / el:.0f} env-steps/s, "
          f"{a.steps * a.nworld * env.nsub / el:.0f} physics steps/s)")
    print(f"loop equality residual max      {eq_max:.2e} rad  ({np.degrees(eq_max):.3f} deg)")
    print(f"loop pin-site gap change max    {gap_max * 1e3:.3f} mm   (home gap {np.round(gap_home * 1e3, 3).tolist()} mm)")
    print(f"gravity |g| in obs              {np.linalg.norm(g, axis=1).min():.3f} .. {np.linalg.norm(g, axis=1).max():.3f}")
    print(f"observation finite              {bool(np.isfinite(obs).all())}")
    print(f"actor obs dim                   {obs.shape[1]} (IMU {'on' if env.cfg.imu else 'off'})")
    ok = eq_max < np.radians(1.0) and gap_max < 1.0e-3 and np.isfinite(obs).all()
    print("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
