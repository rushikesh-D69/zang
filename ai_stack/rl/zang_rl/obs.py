"""Actor observation: ONLY signals the real robot has. Shared by training, sim2sim and (re-implemented in C) the P4.

Two layouts, chosen by EnvCfg.imu (must match the hardware the policy runs on):

  imu = False  (DEFAULT: hardware v0.0.1 has no IMU), 50 values
  imu = True   (future board revision with an IMU), 56 values = the IMU block first, then the same 50

  block        size  units / source on the robot
  gyro         3     rad/s, IMU frame                                          (imu only)
  gravity      3     unit vector in the IMU frame, from the IMU attitude filter (imu only)
  q            15    rad, joint angle minus home, wire order (servo-bus position read, after calibration)
  dq           15    rad/s, (q_t - q_{t-1}) / ctrl_dt from two successive bus reads
  last_action  15    previous policy output (unitless, before scaling)
  command      3     vx m/s, vy m/s, wz rad/s (zeros for the stand task)
  phase        2     sin, cos of the gait phase (zeros for the stand task)

Action: 15 values, wire order. Joint target = home + action_scale * action, clipped to the joint range, sent
with ZLink JOINT_TARGETS (move_ms = ctrl period).
"""
from __future__ import annotations

import numpy as np

N_JOINTS = 15
BLOCKS_BASE = [("q", 15), ("dq", 15), ("last_action", 15), ("command", 3), ("phase", 2)]
BLOCKS_IMU = [("gyro", 3), ("gravity", 3)]


def layout(imu: bool) -> dict:
    """name -> slice for the chosen layout."""
    out, i = {}, 0
    for name, n in (BLOCKS_IMU if imu else []) + BLOCKS_BASE:
        out[name] = slice(i, i + n)
        i += n
    return out


def obs_dim(imu: bool) -> int:
    return sum(n for _, n in (BLOCKS_IMU if imu else []) + BLOCKS_BASE)


def gravity_from_quat(quat, xp=np):
    """World gravity direction (0,0,-1) expressed in the frame with orientation quat (w,x,y,z). Shape (...,3)."""
    w, x, y, z = (quat[..., i] for i in range(4))
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return -xp.stack([r20, r21, r22], -1)


def yaw_from_quat(quat, xp=np):
    w, x, y, z = (quat[..., i] for i in range(4))
    return xp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def build_actor_obs(q_rel, dq, last_action, command, phase, gyro=None, gravity=None, xp=np):
    """IMU-free layout unless gyro and gravity are given. xp = numpy or jax.numpy."""
    parts = [] if gyro is None else [gyro, gravity]
    return xp.concatenate(parts + [q_rel, dq, last_action, command, phase], -1).astype(xp.float32)
