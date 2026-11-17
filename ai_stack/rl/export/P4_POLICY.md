# Running a Zang policy on the ESP32-P4

This is for the firmware team. The files are `export/p4/zang_policy.{c,h}` plus a generated
`logs/<env>/<run>/p4/policy_weights.h`.

## What the network is

- An MLP, 50 → 128 → 128 → 15 (no IMU, hardware v0.0.1) or 56 → 128 → 128 → 15 (with IMU). The hidden layers use ELU.
  Observation normalisation is folded into the header (`zp_obs_mean`, `zp_obs_inv_std`).
- Each call costs about 25 k multiply-adds and 256 `expf`. The P4's HP cores (RISC-V, up to 400 MHz) have a single-precision
  FPU, so plain float C needs well under 1 ms. No esp-dl or quantisation is needed at this size, and the float path
  keeps it bit-compatible with training. `export_policy.py` compiles the C file on the PC with gcc and checks it
  against onnxruntime. The largest measured difference so far is 3.8e-6.
- int8 via esp-dl only makes sense if the network grows by 10× or more. If it does, quantise from `policy.onnx` with esp-dl's
  quantisation tool and re-run the sim2sim check with the quantised model first.

## The 50 Hz loop (one FreeRTOS task, pinned to a core, priority above Wi-Fi/UI)

```
every 20 ms (esp_timer or a periodic task with vTaskDelayUntil):
  1. q      = servo-bus position read of the 15 joints -> model radians (calibration: sign, zero, ticks/rad)
  2. dq     = (q - q_prev) / 0.020
  3. obs    = [ (gyro, gravity if ZP_HAS_IMU) , q - zp_home_q, dq, last_action, command(3), phase(2) ]
  4. zang_policy_run(obs, action);  last_action = action
  5. target_i = clamp(zang_policy_target(i, action_i), lower_i, upper_i)
  6. hand the 15 targets to the same limiter + servo writer that ZLink JOINT_TARGETS uses (move_ms = 20)
```

- **Order and units are the contract.** Joints are in wire order (`ai_stack/config/robot.json`), angles in radians in
  the model frame, and the observation block order is the one in `zang_rl/obs.py`. A sign or offset error in the servo
  calibration shows up as a fall. Calibrate before running any policy.
- **Timing.** The policy was trained with a command latency of 2–24 ms and a position-read delay of 4–30 ms. If the bus
  read of 15 servos takes longer than that, either speed up the bus (sync-read, higher baud, two UARTs) or widen
  `NoiseCfg.q_delay_s` / `RandomizationCfg.latency_s` and retrain. Measure the real loop and report it back.
- **Command source.** The host can send `command` (vx, vy, wz) over ZLink. The policy itself runs on the robot, so
  the 50 Hz loop never depends on Wi-Fi.
- **Safety.** The firmware's last-line limiter (joint range, speed cap, stale-frame hold) stays in front of the
  servos. Also stop the policy and hold if `|q - target|` stays large (more than about 20°) for more than 0.3 s, which
  means a stall or a fall.
