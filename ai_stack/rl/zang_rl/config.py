"""Configuration for the Zang RL environments.

Every number tagged PLACEHOLDER is not measured on the robot yet. The README lists how to measure each one.
Units: SI (m, kg, s, rad) unless a name says otherwise.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
SERVO_JSON = os.path.join(os.path.dirname(HERE), "assets", "servo_ds_r003g.json")

Range = Tuple[float, float]


@dataclasses.dataclass
class ServoCfg:
    """Position-servo model (DS-R003G), applied through the MJCF position actuators.

    This is the same parameter set a BAM-style identification exports for MuJoCo (kp, damping, armature,
    frictionloss, force limit) plus what the bus and firmware add on top (latency, interpolation over the
    command period, speed cap, calibration error). Values come from assets/servo_ds_r003g.json.
    """
    kp: float = 8.0                 # N*m/rad      PLACEHOLDER (zang_description/tools/build.py)
    damping: float = 0.05           # N*m*s/rad    PLACEHOLDER
    armature: float = 0.005         # kg*m^2       PLACEHOLDER (reflected rotor inertia)
    frictionloss: float = 0.01      # N*m          PLACEHOLDER
    force_limit: float = 2.0        # N*m          PLACEHOLDER (stall torque at the battery voltage)
    vmax: float = 6.0               # rad/s        PLACEHOLDER (no-load speed)
    interp: str = "ramp"            # "ramp": DS move command with time = control period (firmware default)
                                    # "step": target jumps at once
    identified: bool = False        # set true by the json once a real identification is pasted in

    @classmethod
    def load(cls, path: str = SERVO_JSON) -> "ServoCfg":
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            raw = json.load(f)
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in fields})


@dataclasses.dataclass
class NoiseCfg:
    """Sensor model for what the real robot reads. All PLACEHOLDER until the board and bus are built."""
    enabled: bool = True
    # servo bus position read
    q_quant: float = 0.00115        # rad, 270 deg / 4096 ticks (Zing's measured DS servo)
    q_std: float = 0.004            # rad white noise
    q_delay_s: Range = (0.004, 0.030)   # bus read latency: 15 servos polled at 115200 baud is ~30 ms worst case
    # IMU
    gyro_std: float = 0.03          # rad/s
    gyro_bias: float = 0.03         # rad/s, per episode, uniform +-
    gravity_std: float = 0.02       # rad, attitude-filter error on the gravity estimate
    imu_tilt: float = 0.03          # rad, per-episode mounting misalignment, uniform +-


@dataclasses.dataclass
class RandomizationCfg:
    """Domain randomisation, resampled per world on every reset."""
    enabled: bool = True
    total_mass_scale: Range = (0.85, 1.35)   # CAD mass is 1.34 kg; no scale reading yet
    body_mass_scale: Range = (0.85, 1.15)    # per body, on top of the total
    com_shift: float = 0.006                 # m, pelvis and chest CoM, uniform +- per axis
    foot_friction: Range = (0.5, 1.2)
    kp_scale: Range = (0.75, 1.25)
    force_scale: Range = (0.7, 1.15)
    damping_scale: Range = (0.6, 1.6)
    frictionloss_scale: Range = (0.5, 2.5)
    armature_scale: Range = (0.7, 1.5)
    latency_s: Range = (0.002, 0.024)        # command latency: host/MCU loop + bus write
    calib_offset: float = 0.035              # rad (2 deg) servo zero error, per joint, uniform +-
    loop_coupling: float = 0.01              # parallelogram coupling slope 1 +- this (pin spacing error)


@dataclasses.dataclass
class PushCfg:
    enabled: bool = True
    interval_s: Range = (2.0, 5.0)
    max_vel_xy: float = 0.6         # m/s kick on the pelvis (1.34 kg: up to 0.8 N*s). Measured with sim2sim:
                                    # home-hold never falls below 0.45 m/s and falls 1/3 of the time at 0.6 m/s,
                                    # so smaller pushes teach no recovery.
    max_ang_vel: float = 0.5        # rad/s


@dataclasses.dataclass
class CommandCfg:
    vx: Range = (-0.10, 0.20)       # m/s; small robot, legs ~12 cm
    vy: Range = (-0.06, 0.06)
    wz: Range = (-0.4, 0.4)         # rad/s; no hip yaw, so turning comes from foot slip / roll stepping only
    zero_prob: float = 0.15
    resample_s: Range = (4.0, 8.0)


@dataclasses.dataclass
class EnvCfg:
    task: str = "stand"             # "stand" | "joystick"
    nworld: int = 32                # ~25 MB per world (own MjModel copy for DR); keep total RAM low
    nthread: int = 0                # 0 = os.cpu_count()
    ctrl_dt: float = 0.02           # 50 Hz policy
    episode_s: float = 20.0
    action_scale: float = 0.3       # rad per unit action, around the home pose
    gym_action_gain: float = 2.0    # Gymnasium env only: action space is [-1, 1], times this gain (+-0.6 rad)
    imu: bool = False               # hardware v0.0.1 has NO IMU -> IMU-free actor observation (obs.py)
    imu_body: str = "chest"         # where a future IMU would sit (sim site; the critic always sees true IMU signals)
    collisions: str = "feet"        # "feet" | "legs" | "all"
    gait_period_s: float = 0.6
    swing_height: float = 0.02      # m
    min_height: float = 0.14        # pelvis z for termination (home 0.23)
    max_tilt: float = 0.8           # rad
    init_noise_q: float = 0.05      # rad
    init_yaw: float = 3.141592653589793   # rad, initial heading uniform in +-this
    servo: ServoCfg = dataclasses.field(default_factory=ServoCfg.load)
    noise: NoiseCfg = dataclasses.field(default_factory=NoiseCfg)
    dr: RandomizationCfg = dataclasses.field(default_factory=RandomizationCfg)
    push: PushCfg = dataclasses.field(default_factory=PushCfg)
    command: CommandCfg = dataclasses.field(default_factory=CommandCfg)
    reward_scales: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.reward_scales:
            self.reward_scales = dict(STAND_REWARDS if self.task == "stand" else JOYSTICK_REWARDS)


STAND_REWARDS = {
    "alive": 0.5,
    "upright": 1.0,
    "height": 1.0,
    "pose": 0.5,
    "still": 0.5,            # low pelvis linear/angular velocity
    "feet_contact": 0.3,
    "foot_slip": -2.0,
    "action_rate": -0.02,
    "torque": -0.002,
    "joint_limit": -2.0,
    "termination": -50.0,
}

JOYSTICK_REWARDS = {
    "alive": 0.25,
    "track_lin_vel": 2.0,
    "track_ang_vel": 0.75,
    "upright": 0.5,
    "height": 0.5,
    "pose": 0.15,
    "gait_contact": 0.5,
    "swing_height": 0.5,
    "foot_slip": -1.0,
    "lin_vel_z": -1.0,
    "ang_vel_xy": -0.05,
    "action_rate": -0.02,
    "torque": -0.002,
    "joint_limit": -2.0,
    "termination": -50.0,
}
