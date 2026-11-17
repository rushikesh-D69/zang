# Sim-to-real pipeline audit (2026-11-17)

This audit follows every link from the CAD to a moving servo. For each link it states what checks it and what is
still a placeholder. Status values: **verified** means checked automatically; **sim-only** means checked against the
emulator and twin but not on hardware yet; **placeholder** means the value needs a measurement.

| # | Link | What carries it | Status | Evidence / next action |
|---|---|---|---|---|
| 1 | CAD → model | `zang_description` (STEP → MJCF/URDF, joints from geometry, parallelogram loop equalities) | verified (geometry) / **placeholder** (dynamics) | IK tests match FK in MuJoCo; loop residual ≤ 0.73° (rl `make check`). Masses and inertias come from CAD densities (1.34 kg). **Weigh the robot and its parts.** |
| 2 | Joint contract | `config/robot.json` → generated `robot_joints.h` | verified | Firmware ctest checks the header matches its generator |
| 3 | Joint limits | `robot.json` lower/upper | **placeholder** (±90°) | Measure the mechanical stops per joint |
| 4 | Calibration (zero, sign, ticks/rev) | firmware NVS, ZLink `CALIB_SET` / `CALIB_GET` / `CALIB_REPORT` | sim-only | The emulator answers CALIB_GET. On the robot, calibrate each servo to the CAD zero pose, then `CALIB_SAVE` |
| 5 | Planner → safety stage | `zang_ai/safety` (limits, CAD-mesh collision, static balance, rate limits, verify) | verified in sim | `tests/test_core.py` (collision projection, limits, speed); all sample replies PASS |
| 6 | Physics twin | `zang_ai/sim/twin.py` (firmware-style ramps, servo latency + speed, MJCF actuators) | sim-only, **placeholder servo** | The servo model (latency 5 ms, 300°/s, kp 8, 2 N·m) is a guess. **Run `make servo-id HOST=<robot> WRITE=1`** |
| 7 | Wire protocol | ZLink v1.1 (`firmware/docs/ZLINK.md`) | verified both sides | 21 golden vectors pass in Python (pytest) and in C (firmware ctest) |
| 8 | Streaming + flow control | `link/client.py` (0.6 s lookahead), STATE `stream_buffered_ms` / `stream_pos_ms` | sim-only | Emulator: buffer and stream position are reported correctly. Over real Wi-Fi (C2 UART link) the jitter is unmeasured; watch `buffered_ms` |
| 9 | Firmware last line | limiter, stale-frame drop, 500 ms watchdog → HOLD | verified (host tests) | Untested on hardware |
| 10 | Servo bus | DS protocol over UART (GPIO47/48 rework) | **blocked** | The v0.0.1 board has no wire from the bus to the P4 (HARDWARE_REVIEW finding 1). The DS-R003G's baud rate, SYNC WRITE and resolution are unconfirmed |
| 11 | Real servo behaviour | — | **unmeasured** | `tools/servo_id.py` fits latency, speed limit, t63, overshoot, deadband and steady error. Checked against the emulator: it recovers the emulator's 229°/s as 236°/s and reaches every step to 0.0° |
| 12 | Whole-motion agreement | `tools/sim2real_replay.py` | sim-only | Plays the same commands on the robot and in the twin and reports per-joint real-vs-sim and real-vs-command RMS and latency, as JSON + CSV. Emulator run: 1.7° real-vs-sim RMS |
| 13 | Balance sensing | IMU | **missing on v0.0.1** | No fall detection. Learned balance policies need an IMU (rl README: with an IMU, about half the falls). Add one on v0.0.2 (I2C GPIO17/18) |
| 14 | Learned policies | `rl/` (Gymnasium/rsl_rl, GPU notebook) → ONNX → C MLP | sim-only | sim2sim across integrators is done. The trained policies don't beat holding the home pose yet; real runs wait on 6, 10, 11 and 13 |
| 15 | Face / display | `faces.json` + one SDF algorithm in Python, JS and C | verified in sim | Golden-image test in pytest; the C port is tested against the same goldens |
| 16 | Audio ↔ motion sync | pre-roll (`audio_offset`), stream clock | sim-only | On the robot, measure audio start vs first frame (a clap plus a high-speed phone video) |

## Order of work on the real robot

1. Do the GPIO47/48 servo-bus rework (link 10). Bring up one servo with `servo_bus` on the bench.
2. Calibrate the zero, sign and ticks/rev of all 15 servos (link 4). Measure the mechanical stops (link 3).
3. `make servo-id HOST=zang.local JOINT=left_elbow WRITE=1`, with an unloaded joint and with a known pendulum load.
   Then copy the values into the MJCF actuators (`zang_description/tools/build.py`) and `rl/assets/servo_ds_r003g.json`.
4. Weigh the robot and its main parts, and update the model masses (link 1).
5. `make sim2real HOST=zang.local`: arms only first, with the robot on a stand. Target: real-vs-sim RMS < 2°. If
   real-vs-sim stays much larger than real-vs-command, fix the twin (steps 3 and 4) and repeat.
6. Then stand on the floor with a slack safety string; co-speech motion only (legs ±4°).
7. Only after an IMU exists: learned balance policies.
