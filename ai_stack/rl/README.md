# Zang RL: learned balance and walking

This is the training ground for future learned controllers:
- **ZangStand**: stand and recover from pushes.
- **ZangJoystick / ZangWalk**: velocity-tracking walk.

It reads the robot from `zang_description/mjcf/` and the joint order from `ai_stack/config/robot.json`, and never edits either.

There are three front ends over **one** task definition (`zang_rl/core.py`), so they cannot drift:

| front end | file | trainer | use it for |
|---|---|---|---|
| Playground-style vectorised env (CPU) | `zang_rl/env.py` | rsl_rl PPO (`train.py`) | **default on the PC**: all worlds in C threads |
| Gymnasium `MujocoEnv`: `ZangStand-v0`, `ZangWalk-v0` | `zang_rl/gym_env.py` | Stable-Baselines3 PPO (`train_sb3.py`) | Gymnasium/SB3 tooling, other algorithms |
| MJX (JAX) env, thousands of worlds on a GPU | `zang_rl/mjx_env.py` | brax PPO (`train_colab_gpu.ipynb`, `zang_rl/brax_train.py`) | **free Colab T4**: long walking runs |

> **Status: infrastructure, not a deployable policy.** The whole pipeline is built and tested: env → PPO →
> checkpoint → ONNX → P4 C MLP, plus sim2sim and the Colab notebook. Only smoke and short runs were trained, and
> **none of the trained policies beats simply holding the home pose yet** (see "Results so far"). The physical parameters
> are placeholders until the measurements in "Closing the gap" are done.

## Quick start (local PC)

```bash
cd ai_stack/rl
make setup            # once: .venv layered on ZingAIStack/.venv-playground (no second torch)
make check            # loop closure + sensor sanity (PASS/FAIL)
make check-gym        # gymnasium env_checker on ZangStand-v0 / ZangWalk-v0
make smoke            # 5 rsl_rl iterations + ONNX/C export (~1 min)
make train-stand      # rsl_rl, 3000 iterations, about 1 h here -> logs/ZangStand/<run>/
make train-sb3        # SB3 PPO on ZangStand-v0, 5 M steps, about 2 h here -> logs/sb3/ZangStand-v0/<run>/
make sim2sim RUN=logs/ZangStand/<run>/
make bundle           # zang_rl_colab_bundle.zip (0.9 MB) for the Colab GPU notebook
make tensorboard
```

```python
import gymnasium as gym, zang_rl                 # importing zang_rl registers the ids
env = gym.make("ZangStand-v0")                   # 50-value IMU-free observation, actions in [-1, 1]
env = gym.make("ZangStand-v0", imu=True)         # future board with an IMU (56 values)
env = gym.make("ZangWalk-v0")
```

## Colab GPU training (`train_colab_gpu.ipynb`)

The CPU backends are limited by MuJoCo stepping, so a free Colab session with 2 CPU cores is slower than this PC. The GPU path
is different: `zang_rl/mjx_env.py` steps thousands of worlds at once with MJX on a **Colab T4**, and brax PPO trains on them.

**Workflow**
1. `make bundle` builds `zang_rl_colab_bundle.zip` (about 0.9 MB, 101 files). Put it on Google Drive once, or upload it each session.
2. Open `train_colab_gpu.ipynb` in Colab, choose *Runtime > Change runtime type > T4 GPU*, set `TASK` (`stand` or `walk`),
   `NUM_ENVS`, `TOTAL_STEPS` and `SEED` in the Step 5 cell, and run all.
3. At the end the notebook downloads one results zip: `policy_brax.npz`, `policy.onnx`, `policy_meta.json`,
   `p4/policy_weights.h`, the learning curves, the configs, the sim2sim report and the last checkpoint.

**What the notebook does, in order:** check the GPU (fails clearly without one), upload and unzip the bundle, verify every file against
`MANIFEST.json` (sha256, stops on any mismatch), install the pinned packages from `requirements_colab.txt` (`jax[cuda12]` 0.12.0.1,
`mujoco` and `mujoco-mjx` 4.0.1, `brax` 0.14.2, `flax` 0.12.12, ...), build the MJX env and **assert that it equals the CPU
Gymnasium env** (observation size and block names, action size, joint order from `robot.json`, home pose, sensor layout,
reward terms, whole config, same start state), measure throughput, train with brax PPO, plot the learning curves, export ONNX and the
C header with `export_policy.py`, evaluate the exported ONNX with `sim2sim.py` in CPU MuJoCo (with the home-pose baseline),
and zip the results.

**Same task as the CPU envs.** `mjx_env.py` calls the same `core.py` functions with `jax.numpy`: the 50-value IMU-free observation,
the 15 actions (target = home + 0.3 rad x 2 x clip(a, -1, 1)), rewards, domain randomisation, pushes and the servo command path.
Differences: stepping is `mjx.step` (float32, implicit-fast, 2 ms, 10 substeps), the feet are boxes against the floor plane with no
self collision (the CPU envs use the same box feet by default, so the contact geometry is the same), and the closed parallelogram loops are the same
joint-coupling equalities. Auto-reset happens inside the env, so each episode gets fresh randomisation, calibration and servo buffers.
`tests/test_backend_parity.py` (run it on the PC) compares numpy and jax on identical inputs and an MJX rollout with a CPU one.

**brax PPO setup.** Asymmetric actor-critic (the actor sees the IMU-free observation; the critic also gets the privileged state),
actor 50-128-128-15 with ELU (about 25 k MACs, the size the P4 runs), critic 256-256-128, Gaussian policy with log-std
(not tanh-normal, because the P4 C MLP has no tanh: the output goes through `clip(-1, 1) x 2` exactly as in `export_policy.py`
for SB3), fixed learning rate 3e-4, observation normalisation folded into the exported ONNX.

**Resume.** Every `CHECKPOINT_EVERY` steps the notebook writes an atomic numpy checkpoint to `OUT_ROOT/RUN_NAME` (Google Drive when
mounted) and keeps the newest 3. Run all with the same `RUN_NAME` to continue after a disconnect. Adam moments are not restored (the weights, the
value network and the observation normaliser are).

**Time (ESTIMATES: the notebook was not run on a T4 when it was written).** The notebook prints a measured env-steps/s
for your runtime in Step 4 and an ETA during training. Rough expectation from public MJX benchmarks: 5 k to 25 k env steps/s on a T4,
so 50 M stand steps take about 0.5 to 3 h and 150 M walk steps about 2 to 8 h, plus a few minutes of XLA compile for every session.
For comparison, the CPU runs here measure about 2.7 k env steps/s in training on 4 cores (rsl_rl).

**What the zip holds** (`make_bundle.py`): `zang_rl/` without the rsl_rl-only modules (`env.py`, `registry.py`), `robot.json`,
`servo_ds_r003g.json`, the RL MJCF (`scene.xml`, `zang.xml`), the collision meshes (only `sim2sim.py` reads them, for full-contact
physics; training reads no mesh at all, because `robot.build()` deletes the visual geoms and the box feet replace the foot meshes),
`export_policy.py`, `sim2sim.py` and the P4 C sources that `export_policy.py` compiles to check the header,
`requirements_colab.txt` and `MANIFEST.json`. No visual meshes (23 MB). The zip is reproducible (fixed timestamps).

**Kaggle / other GPUs:** the notebook only uses `google.colab` for the upload, Drive and download; set `UPLOAD_MODE` and the paths in
Step 2 by hand elsewhere. The `jax[cuda12]` pin needs a CUDA 12 capable driver.

## Layout

| path | what |
|---|---|
| `zang_rl/core.py` | **the task**: randomisation, initial state, servo command path, observation, rewards, terminations, commands, pushes (numpy, no torch) |
| `zang_rl/robot.py` | builds the RL model from the canonical MJCF with `MjSpec`. Adds the IMU site, sole sites and sensors, applies collision filtering, strips visual meshes, and holds the loop checks. |
| `zang_rl/config.py` | env, servo, noise, randomisation, push and command config. Every placeholder is tagged. |
| `zang_rl/obs.py` | the actor observation contract: training, sim2sim and the P4 C code all follow it |
| `zang_rl/env.py` | rsl_rl VecEnv on `mujoco.rollout` (one MjModel per world) |
| `zang_rl/mjx_env.py` | MJX (JAX) brax env, same task through `core.py` with `jax.numpy`; auto-reset inside the env |
| `zang_rl/brax_train.py` | brax PPO glue: network factory, numpy checkpoints (atomic, resumable), actor export to `.npz` |
| `zang_rl/gym_env.py` | Gymnasium `MujocoEnv` subclass, registered in `zang_rl/__init__.py` |
| `zang_rl/registry.py` | rsl_rl task ids and runner configs (Playground's config dataclasses) |
| `train.py`, `train_sb3.py` | the two trainers. Both export at the end. |
| `sim2sim.py` | an independent `mj_step` loop written like the P4 control loop, run under different physics |
| `export_policy.py` | rsl_rl `.pt`, SB3 `.zip` or brax `policy_brax.npz` → `policy.onnx`, `policy_meta.json`, `p4/policy_weights.h`, plus a C-vs-ONNX check |
| `export/p4/`, `export/P4_POLICY.md` | the P4 C MLP and how to run it at 50 Hz |
| `train_colab_gpu.ipynb`, `make_bundle.py`, `requirements_colab.txt` | Colab GPU training: notebook, the zip it needs, its pinned packages |
| `tests/test_backend_parity.py` | numpy vs jax core maths, layouts of the three envs, MJX vs CPU rollout (run on the PC) |
| `check_env.py` | parallelogram-loop and sensor verification |
| `assets/servo_ds_r003g.json` | DS-R003G servo parameters. **PLACEHOLDER**. |

## Why a custom `MujocoEnv` subclass, not `Humanoid-v5` + `xml_file`

`Humanoid-v5` drives **torque** actuators and observes 348 privileged values (cinert, cvel, cfrc_ext). Zang has
**position servos** behind a bus latency, **no IMU**, and closed parallelogram loops. The observation has to be what the
robot can read. Each action has to go through the servo command path (latency, ramp, speed cap, zero error) at
every physics substep, and a Reward/Observation wrapper cannot change that.

`ZangGymEnv` overrides `_initialize_simulation()` so the model comes from `robot.build`. It also overrides
`_set_action_space()`, `reset_model()` and `step()`. These names were checked against the installed gymnasium 1.4.0
`envs/mujoco/mujoco_env.py`. `gymnasium.utils.env_checker.check_env` passes for both ids. The only warnings left
are the expected ones for an unbounded observation Box.

## Backends: CPU on the PC, MJX on a cloud GPU

The installed Playground 1.0.1 on this PC is built on torch and MuJoCo Warp, and the local GPU is a **GTX 1050 Ti (sm_61)**: the installed
torch (cu130) and Warp (CUDA 13) wheels need sm_75 or newer and fail with "no kernel image". So on the PC the fast path is
`mujoco.rollout`: about 5.5k env steps/s (55k physics steps/s) with 32 worlds on 4 cores, about 1 GB RAM.

For large walking runs, the **MJX backend** (`mjx_env.py`, pure-JAX MJX, not MuJoCo Warp) runs on a Colab T4 (sm_75). It is only used
from the notebook above. `ai_stack/rl/.venv` has jax, brax and mujoco-mjx installed (CPU jax) so the env and the parity test can be
traced and tested on the PC, but training on this PC's GPU is not possible.

## What is modelled (realism over reward hacking)

**Closed loops.** The source MJCF closes each half-leg parallelogram with joint-coupling equalities (`mjEQ_JOINT`),
and those are kept. `check_env.py` verifies them under random actions, pushes and full randomisation:
- equality residual at most **0.73°** (soft-constraint stretch under impacts);
- true pin-site loop gap at most **0.63 mm** beyond its 0.2 mm home value. The CAD pin spacing is not an ideal parallelogram;
  a randomised ±1% coupling slope stands in for that error.

Under sim2sim (1 ms RK4, full contacts) the residual stays at or below 0.23° and the gap at or below 0.24 mm.

**Servo (DS-R003G).** Position actuators use kp, force limit, damping, armature and frictionloss from
`assets/servo_ds_r003g.json`. Every command passes through the real command path:
- latency of 2–24 ms;
- a linear ramp over the 20 ms period, like the DS move command with time = period;
- a speed cap;
- a ±2° servo zero error.

This parameter set is exactly what a BAM-style identification or `mujoco.sysid` exports. A torque-level BAM model would
need per-substep feedback. The hook for it is `core.servo_profile` + `ServoCfg`. Add it only if identification shows the
position model is not enough.

**Sensors (the actor sees only what the robot has).** At 50 Hz:
- servo-bus joint positions with a 4–30 ms read delay, 270°/4096 quantisation and noise;
- `dq` estimated from two successive reads;
- the last action;
- the command and the gait phase.

That is **50 values, no IMU**. With `imu=True`, gyro (noise, bias) and gravity (noise, mounting tilt) come first (56
values). The **critic** is asymmetric and used in training only: it gets noise-free joints, the true IMU, base velocity, height,
foot contact and height, and the randomisation parameters.

**Domain randomisation** (per world, every reset):
- total mass 0.85–1.35× (the 1.34 kg is CAD-derived), and each body ±15%;
- pelvis/chest CoM ±6 mm;
- foot friction 0.5–1.2;
- kp 0.75–1.25×, force 0.7–1.15×, damping, armature, frictionloss;
- latency, calibration offsets, loop coupling;
- pushes every 2–5 s up to **0.6 m/s** on the pelvis.

The push size is measured: holding the home pose never falls below 0.45 m/s and falls 3 in 8 times at 0.6 m/s.

**Hardware respected.** There is no hip yaw, so turning comes only from stepping and the yaw command is small (±0.4 rad/s).
`knee_pitch` is the lower-leg servo. Actions are 15 servo targets in wire order, as offsets from the home crouch:
target = home + 0.3 rad × action. The Gym action space is [-1, 1], times a gain of 2.

## Results so far (short runs: read as trends, not as capability)

All runs: ZangStand, 150 PPO iterations each (460 k env steps, about 3–4 min). Evaluated with `sim2sim.py`, 8 episodes × 10 s,
a push every 2.5 s, sensor noise on, nominal parameters. Each cell shows "train-like physics / sim2sim physics".

| policy | fall rate @ 0.45 m/s | @ 0.6 m/s | @ 0.75 m/s | joint tracking RMS |
|---|---|---|---|---|
| home pose held (zero action) | **0% / 0%** | 38% / 50% | 100% / 75% | 0.7° |
| learned, no IMU (adaptive LR) | 100% / 88% | 100% / 100% | 100% / 100% | 9–11° |
| learned, no IMU (fixed LR 3e-4) | 88% / 50% | 100% / 62% | n/a | 7–8° |
| learned, **with IMU** (adaptive LR) | **38% / 50%** | 50% / 75% | 100% / 75% | 5–6° |

- **IMU vs no IMU, same budget and settings:** the IMU policy falls about half as often at 0.45 and 0.6 m/s, and tracks
  its own commands about 2× more tightly. With 8 episodes per cell, the per-cell rates are noisy, but the direction holds at every push level.
- **No learned policy beats the stiff home pose yet.** At 150 iterations the policies are still noisy, and this robot
  is passively very stable when it holds still: a low crouch, wide feet, stiff servos. A useful stand policy needs
  thousands of iterations (about 1 h locally).
- **sim2sim transfer:** at the same push, falls in the 1 ms RK4, full-contact physics are within about ±25 points of the
  training physics in either direction. That is as expected for policies this short, with no systematic collapse.

**Training notes.** With rsl_rl's adaptive (KL) schedule, the learning rate sat at its 1e-5 floor for whole runs, and
learning stalled. The default is now a fixed 3e-4, which learned faster in the same budget (row 3 vs row 2).

## IMU: hardware v0.0.1 has none, and balance will very likely need one

Without an IMU, the policy only knows the robot's tilt through joint-angle side effects (servo sag under load,
foot-contact changes). These are delayed and ambiguous, so it notices a push late. With the same short training
budget, adding the IMU halved the fall rate at moderate pushes and halved tracking error (table above). Push recovery
and walking need tilt and angular rate within a control period or two. **Plan an IMU for the next board revision.**

- **Part:** **QMI8658C** (QST, 6-axis), on I2C at 400 kHz. Zing already uses it, so the driver in
  `zing3.0/components/modules/qmi8658/` ports directly. Wire INT1 to a P4 GPIO for data-ready.
  Alternative: **ICM-42688-P** (TDK) on SPI, for lower gyro noise and latency.
- **Placement:** on the main board near the torso centre, rigidly mounted. Record its orientation (`EnvCfg.imu_body`, chest by default).
- **Rate and filter:** gyro and accelerometer at 200 Hz or more. Use a complementary or Mahony filter on the P4 for the gravity vector.
- **Software is ready:** the `imu=True` layout, its noise model, `ZangStandIMU` and `gym.make(..., imu=True)` exist,
  and `policy_weights.h` carries `ZP_HAS_IMU`. Adding the chip needs only a config switch and a retrain.

## Policy export and the P4

`export_policy.py` takes an rsl_rl `.pt` or an SB3 `.zip` (with `vecnormalize.pkl`). It writes:
- `policy.onnx`: normalisation folded in; the output is the action in the shared convention;
- `policy_meta.json`;
- `p4/policy_weights.h`.

It then compiles `export/p4/zang_policy.c` with gcc and compares it with onnxruntime on 200 random observations. The max
|diff| was 3.8e-6 to 5.7e-6 for rsl_rl and 2e-5 to 6e-5 for SB3. The actor is about 25 k MACs, which is plain float C on the P4 in well under 1 ms.
Details: [export/P4_POLICY.md](export/P4_POLICY.md).

## What is realistic now, and what is placeholder

| item | status |
|---|---|
| kinematics, joint axes, parallelogram coupling | from CAD, verified (IK/test_ik.py, check_env.py) |
| masses, inertias, CoM | **CAD-derived**; 1.34 kg not weighed |
| servo kp, torque, speed, friction, armature | **placeholders** (build.py), randomised widely |
| latency, bus read delay | **assumed** from the 115200-baud single-servo protocol, randomised |
| foot friction, sensor noise | **assumed**, randomised |
| IMU | **absent** on v0.0.1; modelled as an option |

## Closing the gap: what to measure on the real robot

1. **Servo identification (most important).** Use a DS-R003G pendulum bench. Command steps of 10–60° at several loads and
   both move times, and log position at 200 Hz or more. Fit kp, damping, frictionloss, armature and force limit with `mujoco.sysid` or BAM.
   Put the result in `assets/servo_ds_r003g.json`, set `identified: true`, and narrow the randomisation ranges.
2. **Mass and CoM.** Weigh the robot and each limb assembly, and find the CoM height by balancing the robot on an edge. Then update
   `zang_description/tools/build.py` and narrow `total_mass_scale`.
3. **Loop timing on the P4.** Measure command-to-motion latency and the time to read all 15 servos, and set `latency_s` and `q_delay_s` from them.
4. **Calibration error.** Measure each joint's zero error after calibration, and set `calib_offset`.
5. **Foot friction.** Do a tilt-plane test on the real floor, and set `foot_friction`.
6. **Replay test.** Run the same scripted motion on the robot and in `sim2sim.py`, and compare tracking RMS and the time to slip or fall.
   When they agree, try a stand policy on the robot with a slack safety string.
